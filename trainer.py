import os
import time
import warnings

warnings.filterwarnings('ignore')
import torch as th
from torch import transpose as tr
import torch.nn.functional as F
from torch import nn
import random
from itertools import permutations
from dataset import logger
from torch.nn.utils.rnn import PackedSequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import librosa as audio_lib
from stft_util import STFT, iSTFT, IPD
import math
from collections.abc import Sequence


def create_optimizer(optimizer, params, **kwargs):
    supported_optimizer = {
        'sgd': th.optim.SGD,  # momentum, weight_decay, lr
        'rmsprop': th.optim.RMSprop,  # momentum, weight_decay, lr
        'adam': th.optim.Adam,  # weight_decay, lr
        'adamW' : th.optim.AdamW,
        'adadelta': th.optim.Adadelta,  # weight_decay, lr
        'adagrad': th.optim.Adagrad,  # lr, lr_decay, weight_decay
        'adamax': th.optim.Adamax  # lr, weight_decay
        # ...
    }
    if optimizer not in supported_optimizer:
        raise ValueError('Now only support optimizer {}'.format(optimizer))
    if optimizer != 'sgd' and optimizer != 'rmsprop':
        del kwargs['momentum']
    opt = supported_optimizer[optimizer](params, **kwargs)
    logger.info('Create optimizer {}: {}'.format(optimizer, kwargs))
    return opt


def packed_sequence_cuda(packed_sequence, device):
    #if not isinstance(packed_sequence, PackedSequence):
    #    raise ValueError("Input parameter is not a instance of PackedSequence")
    if th.cuda.is_available():
        packed_sequence = packed_sequence.to(device)
    return packed_sequence

class WarmupConstantSchedule(th.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to "init_lr" over `warmup_steps` training steps.
        Keeps learning rate schedule equal to "init_lr". after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            return 1.0

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)


class PITrainer(object):
    def __init__(self,
                 nnet,
                 stft_config,
                #  rir_config,
                 checkpoint="checkpoint",
                 loss='A_SDR',
                 scale_inv=True,
                 optimizer="adam",
                 lr=1e-5,
                 momentum=0.9,
                 weight_decay=0,
                 clip_norm=None,
                 crm=False,
                 min_lr=0,
                 patience=1,
                 factor=0.5,
                 disturb_std=0.0,
                 mvn=True,
                 apply_log=False,
                 IPD_sincos=True,
                 angle_feature_opt=False,
                 gpuid=0):
        # multi gpu
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device('cuda:{}'.format(gpuid[0]))
        self.gpuid = gpuid
        # self.scaler = th.cuda.amp.GradScaler()

        self.stft_config = stft_config
        self.nnet = nnet.to(self.device)
        self.crm = crm
        self.mvn = mvn
        self.apply_log = apply_log
        self.IPD_sincos = IPD_sincos
        self.loss = loss
        self.scale_inv = scale_inv
        logger.info("Network structure:\n{}".format(self.nnet))
        self.optimizer = create_optimizer(
            optimizer,
            self.nnet.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)
        # self.scheduler = th.optim.lr_scheduler.StepLR(self.optimizer, 
        #                                                  step_size=5, 
        #                                                  gamma=0.8, 
        #                                                  verbose=True)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True)
        self.warmup_scheduler = WarmupConstantSchedule(self.optimizer, warmup_steps=1000)
        self.checkpoint = checkpoint
        self.num_spks = nnet.num_spks
        self.clip_norm = clip_norm
        self.disturb = disturb_std
        if self.disturb:
            logger.info("Disturb networks with std = {}".format(disturb_std))
        if self.clip_norm:
            logger.info("Clip gradient by 2-norm {}".format(clip_norm))
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6
        logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        self.location = 'cuda:{}'.format(gpuid[0])
        checkpoint_list = []
        for file in os.listdir(checkpoint):
            if file.endswith(".pkl"):
                checkpoint_list.append(file)
        checkpoint_list_int = [int(a.split('.')[1]) for a in checkpoint_list]
        if not checkpoint_list_int == []:
            max_val = max(checkpoint_list_int)
            max_index = checkpoint_list_int.index(max_val)
            self.checkpoint_file = checkpoint + '/' + checkpoint_list[max_index]
            logger.info("Loaded Pretrained model of {} .....".format(self.checkpoint_file))
            checkpoint_dict = th.load(self.checkpoint_file, map_location=self.location)
            self.nnet.load_state_dict(checkpoint_dict['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            self.start_epoch = checkpoint_dict['epoch'] + 1
        else:
            self.start_epoch = 1
        # self.optimizer.param_groups[0]["lr"] = 9.8e-5
        
        self.stft = STFT(stft_config["frame_length"],stft_config["frame_shift"],stft_config["window"], device=self.device)
        self.inverse_stft = iSTFT(stft_config["frame_length"], stft_config["frame_shift"], stft_config["window"], device=self.device)
        self.angle_feature_opt = angle_feature_opt

        # self.RT_list = rir_config["RT_list"]
        # self.rir_dir = rir_config["rir_dir"]
        # self.batch_rir_mix = RIR_mixer(self.gpuid,**rir_config, num_spks=self.num_spks)

    def extractor(self, input_sizes, mixture, target_rir, target, noise):
        
        input_sizes = th.floor(input_sizes/self.stft_config["frame_shift"] - 1)
        source_attr = {}
        target_attr = {}
        target_attr["spectrogram"] = []
        target_attr["phase"] = [] 
        noise_attr = {}
        if len(mixture.shape) == 3:
            source_attr["spectrogram"], source_attr["phase"] = self.stft(mixture[:,:,0].to(self.device))
            mix_STFT = self.stft(tr(mixture,1,2).to(self.device))
            mix_STFT = tr(convert_complex(mix_STFT[0],mix_STFT[1]),1,3)
            for t in target:
                tmp = self.stft(t.to(self.device))
                target_attr["spectrogram"].append(tmp[0])
                target_attr["phase"].append(tmp[1])
            noise_attr["spectrogram"], noise_attr["phase"] = self.stft(noise[:,:,0].to(self.device))
            mix_feat = apply_cmvn(mix_STFT[:,:,:,0]) if self.mvn else mix_STFT[:,:,:,0]
            if self.crm:
                mix_feat = th.cat([th.real(mix_feat), th.imag(mix_feat)],dim=-1)
            else:
                # mix_feat = th.cat([th.real(mix_feat), th.imag(mix_feat)],dim=-1)
                mix_feat = th.abs(mix_feat)
            if self.apply_log: mix_feat = th.log(th.clamp(mix_feat, min=1.0e-6))
            
            for i in range(1,mix_STFT.shape[-1]):
                ipd = IPD(mix_STFT[:,:,:,0], mix_STFT[:,:,:,i], cos_sin_opt=self.IPD_sincos)
                mix_feat = th.cat([mix_feat, ipd], dim=-1)
        else:
            source_attr["spectrogram"], source_attr["phase"] = self.stft(mixture.to(self.device))
            mix_STFT = self.stft(mixture.to(self.device))
            mix_STFT = tr(convert_complex(mix_STFT[0],mix_STFT[1]),1,2)
            for t in target:
                tmp = self.stft(t.to(self.device))
                target_attr["spectrogram"].append(tmp[0])
                target_attr["phase"].append(tmp[1])
            noise_attr["spectrogram"], noise_attr["phase"] = self.stft(noise.to(self.device))
            mix_feat = apply_cmvn(mix_STFT) if self.mvn else mix_STFT
            mix_feat = th.abs(mix_feat)
            if self.apply_log: mix_feat = th.log(th.clamp(mix_feat, min=1.0e-6))
        nnet_input = packed_sequence_cuda(mix_feat, self.device) if isinstance(
            mix_feat, PackedSequence) else mix_feat.to(self.device)  
        return input_sizes, source_attr, target_attr, noise_attr, nnet_input

    def train(self, dataset, epoch):
        self.nnet.train()
        logger.info("Training...")
        tot_loss_s = tot_loss_n = num_batch = 0
        pbar = tqdm(total=len(dataset), unit='batches', bar_format='{l_bar}{bar:80}{r_bar}{bar:-10b}', colour="YELLOW", dynamic_ncols=True)
        for input_sizes, mixture, target_rir, target, noise in dataset:
            # mixture, target_rir, target, noise = self.batch_rir_mix.RIR_mixing(target, len(input_sizes), train=True)

            extracted_feature = self.extractor(input_sizes, mixture, target_rir, target, noise)
            # input_sizes, source_attr, target_attr, noise_attr, nnet_input, angle_diff, AF = extracted_feature
            input_sizes, source_attr, target_attr, noise_attr, nnet_input = extracted_feature
            num_batch += 1
            pbar.update(1)
            # scheduler learning rate for warm-up
            if epoch == 1: self.warmup_scheduler.step()
            nnet_input = packed_sequence_cuda(nnet_input, self.device) if isinstance(
                nnet_input, PackedSequence) else nnet_input.to(self.device)
            self.optimizer.zero_grad()
            if self.disturb: self.nnet.disturb(self.disturb)

            masks = th.nn.parallel.data_parallel(self.nnet, nnet_input, device_ids=self.gpuid)
            cur_loss_s = self.PIT_loss_spec(masks[:self.num_spks], input_sizes, source_attr, target_attr)
            # cur_loss_s = self.PIT_loss(masks[:self.num_spks], input_sizes, source_attr, target_attr)
            tot_loss_s += cur_loss_s.item() / self.num_spks
            if len(masks) == self.num_spks:
                cur_loss = cur_loss_s / self.num_spks
            else:
                cur_loss_n = self.Noise_loss(masks[self.num_spks], input_sizes, source_attr, noise_attr)
                cur_loss = (cur_loss_s + cur_loss_n) / (self.num_spks + 1)            
                tot_loss_n += cur_loss_n.item()

            cur_loss.backward()
            if self.clip_norm:
                th.nn.utils.clip_grad_norm_(self.nnet.parameters(),
                                            self.clip_norm)
            self.optimizer.step()
            pbar.set_postfix({'Loss': tot_loss_s/num_batch, 'Loss_n': tot_loss_n/num_batch})
        pbar.close()
        return tot_loss_s / num_batch, num_batch


    def validate(self, dataset):
        self.nnet.eval()
        logger.info("Cross Validate...")
        tot_pesq_in = tot_pesq_out = tot_loss_s = tot_loss_n = num_batch = 0
        # do not need to keep gradient
        pbar = tqdm(total=len(dataset), unit='batches', bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}', colour="RED", dynamic_ncols=True)
        with th.no_grad():
            for input_sizes, mixture, target_rir, target, noise in dataset:
                extracted_feature = self.extractor(input_sizes, mixture, target_rir, target, noise)
                input_sizes, source_attr, target_attr, _, nnet_input = extracted_feature
                # input_sizes, source_attr, target_attr, noise_attr, nnet_input, angle_diff, AF = extracted_feature

                num_batch += 1
                pbar.update(1)
                                
                masks = th.nn.parallel.data_parallel(self.nnet, nnet_input, device_ids=self.gpuid)
                # cur_loss_s = self.PIT_loss(masks[:self.num_spks], input_sizes, source_attr, target_attr)
                cur_loss_s = self.PIT_loss_spec(masks[:self.num_spks], input_sizes, source_attr, target_attr)
                tot_loss_s += cur_loss_s.item() / self.num_spks
                # th.cuda.empty_cache() # personal edit
                pbar.set_postfix({'Loss': tot_loss_s/num_batch, 'Loss_n': tot_loss_n/num_batch, 'pesq_in': tot_pesq_in/num_batch, 'pesq_out': tot_pesq_out/num_batch})
        pbar.close()
        
        return tot_loss_s / num_batch, num_batch


    def run(self, train_set, dev_set, num_epoches=20, log_dir='log'):
        with th.cuda.device(self.gpuid[0]):
            writer_src = SummaryWriter(log_dir)
            start_time = time.time()
            init_loss = 0
            if self.start_epoch > 1:
                init_loss, _ = self.validate(dev_set)
            end_time = time.time()
            logger.info("Epoch {:2d}: dev = {:.4f}({:.2f}s)".format(self.start_epoch-1, init_loss,end_time-start_time))
            for epoch in range(self.start_epoch, num_epoches + self.start_epoch):
                valid_loss_best = init_loss
                on_train_start = time.time()
                train_loss_src, train_num_batch = self.train(train_set, epoch)
                on_valid_start = time.time()
                valid_loss_src, valid_num_batch = self.validate(dev_set)
                on_valid_end = time.time()
                logger.info(
                    "Loss(time/mini-batch) \n - Epoch {:2d}: train for source = {:.4f}({:.2f}s/{:d}) |"
                    " dev for source = {:.4f}({:.2f}s/{:d})".format(
                        epoch, train_loss_src, on_valid_start - on_train_start,
                        train_num_batch, valid_loss_src, on_valid_end - on_valid_start,
                        valid_num_batch))
                writer_src.add_scalars("Loss_src",{'train':train_loss_src, 'valid':valid_loss_src}, epoch)
                writer_src.flush()

                self.scheduler.step(valid_loss_src)

                if valid_loss_src < valid_loss_best:
                    valid_loss_best = valid_loss_src
                    save_path = os.path.join(self.checkpoint,
                                            'epoch.{:d}.pkl'.format(epoch))
                    th.save(
                            {
                            'epoch': epoch,
                            'model_state_dict': self.nnet.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss' : valid_loss_src
                            },
                            save_path)
                        
                    
        logger.info("Training for {} epoches done!".format(num_epoches))

    def PIT_loss(self, masks, input_sizes, source_attr, target_attr):

        input_sizes = input_sizes.to(self.device)
        mixture_spect = source_attr["spectrogram"].to(self.device)
        mixture_phase = source_attr["phase"].to(self.device)
        targets_spect = [t.to(self.device) for t in target_attr["spectrogram"]]
        targets_phase = [t.to(self.device) for t in target_attr["phase"]]
        if self.num_spks != len(targets_spect):
            raise ValueError(
                "Number targets do not match known speakers: {} vs {}".format(
                    self.num_spks, len(targets_spect)))

        def A_SDR_loss(permute, eps=1.0e-6):
            loss_for_permute = []
            for s, t in enumerate(permute):
                if self.crm:
                    mask_r, mask_i = th.chunk(masks[s], 2, 2)
                    estim_r, estim_i = mixture_spect * th.cos(mixture_phase), mixture_spect * th.sin(mixture_phase)
                    x = self.inverse_stft(mask_r*estim_r - mask_i*estim_i, mask_r*estim_i + mask_i*estim_r, cplx=True)
                else:
                    x = self.inverse_stft(masks[s]*mixture_spect, mixture_phase, cplx=False)                    
                s = self.inverse_stft(targets_spect[t], targets_phase[t], cplx=False)

                x_zm = x - th.mean(x, dim=-1, keepdim=True)
                s_zm = s - th.mean(s, dim=-1, keepdim=True)
                if self.scale_inv: 
                    s_zm = th.sum(x_zm * s_zm, dim=-1, keepdim=True) / (l2norm(s_zm, keepdim=True)**2 + eps) * s_zm
                utt_loss = - 20 * th.log10(eps + l2norm(s_zm) / (l2norm(x_zm - s_zm) + eps))
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)

        def SA_SDR_loss(permute, eps=1.0e-6):
            s_zm_SA = x_zm_SA = th.tensor([]).to(self.device)
            for s, t in enumerate(permute):
                if self.crm:
                    mask_r, mask_i = th.chunk(masks[s], 2, 2)
                    estim_r, estim_i = mixture_spect * th.cos(mixture_phase), mixture_spect * th.sin(mixture_phase)
                    x = self.inverse_stft(mask_r*estim_r - mask_i*estim_i, mask_r*estim_i + mask_i*estim_r, cplx=True)
                else:
                    x = self.inverse_stft(masks[s]*mixture_spect, mixture_phase, cplx=False)
                s = self.inverse_stft(targets_spect[t], targets_phase[t], cplx=False)                
                x_zm = x - th.mean(x, dim=-1, keepdim=True)
                s_zm = s - th.mean(s, dim=-1, keepdim=True)
                s_zm_SA = th.cat([s_zm_SA, s_zm], dim=-1)
                x_zm_SA = th.cat([x_zm_SA, x_zm], dim=-1)
            if self.scale_inv:
                s_zm_SA = th.sum(x_zm_SA * s_zm_SA, dim=-1, keepdim=True) / (l2norm(s_zm_SA, keepdim=True)**2 + eps) * s_zm_SA
            tau =  1.0e-6
            return - 20 * th.log10((l2norm(s_zm_SA) + eps) / (l2norm(x_zm_SA - s_zm_SA) + tau*(l2norm(s_zm_SA) + eps) ))  * self.num_spks


        pscore = th.stack( [(A_SDR_loss(p) if self.loss == 'A_SDR' else SA_SDR_loss(p) if self.loss == 'SA_SDR' else MSE_loss(p)) 
                            for p in permutations(range(self.num_spks))] )
        min_perutt, _ = th.min(pscore, dim=0)
        num_utts = input_sizes.shape[0]
        return th.sum(min_perutt) / num_utts


    def PIT_loss_spec(self, masks, input_sizes, source_attr, target_attr):
    
        input_sizes = input_sizes.to(self.device)
        mixture_spect = source_attr["spectrogram"].to(self.device)
        targets_spect = [t.to(self.device) for t in target_attr["spectrogram"]]

        def SA_STFT_Mag_SDR_loss(permute, eps=1.0e-6):
            src_SA = mix_SA = th.tensor([]).to(self.device)
            for s, t in enumerate(permute):
                mix = masks[s]*mixture_spect
                src = targets_spect[t]
                src_SA = th.cat([src_SA, src], dim=-1)
                mix_SA = th.cat([mix_SA, mix], dim=-1)
                # return 20 * th.log10( matrix_norm(mix_SA - src_SA) + eps) * self.num_spks
            return - 20 * th.log10( eps + l2norm(l2norm((src_SA))) / (l2norm(l2norm(mix_SA - src_SA)) + eps))  * self.num_spks 
                
        def STFT_Mag_SDR_loss(permute, eps=1.0e-6):
            loss_for_permute = []
            for s, t in enumerate(permute):
                mix = masks[s]*mixture_spect
                src = targets_spect[t]
                # utt_loss = 20 * th.log10( (matrix_norm(mix - src) + eps))
                utt_loss = - 20 * th.log10( eps + l2norm(l2norm((src))) / (l2norm(l2norm(mix - src)) + eps))
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)

        pscore = th.stack( [ ( SA_STFT_Mag_SDR_loss(p) if self.loss == 'SA_SDR' else STFT_Mag_SDR_loss(p) )  
                                for p in permutations(range(self.num_spks))] )
        min_perutt, _ = th.min(pscore, dim=0)
        num_utts = input_sizes.shape[0]
        return th.sum(min_perutt) / num_utts



    def Noise_loss(self, mask, input_sizes, source_attr, target_attr):

        input_sizes = input_sizes.to(self.device)
        mixture_spect = source_attr["spectrogram"].to(self.device)
        mixture_phase = source_attr["phase"].to(self.device)
        targets_spect = target_attr["spectrogram"].to(self.device)
        targets_phase = target_attr["phase"].to(self.device)

        def MSE_loss():
            if self.crm:
                mask_r, mask_i = th.chunk(mask, 2, 2)
                mixture_r = mixture_spect * th.cos(mixture_phase)
                mixture_i = mixture_spect * th.sin(mixture_phase)
                sig_r = targets_spect * th.cos(targets_phase)
                sig_i = targets_spect * th.sin(targets_phase)
                loss_r =  th.sum(th.sum(th.pow(th.abs((mask_r*mixture_r - mask_i*mixture_i) - sig_r), 2), -1),-1) / input_sizes
                loss_i =  th.sum(th.sum(th.pow(th.abs((mask_r*mixture_i + mask_i*mixture_r) - sig_i), 2), -1),-1) / input_sizes
                utt_loss = loss_r + loss_i
            else:
                utt_loss = th.sum(th.sum(th.pow(th.abs(mask*mixture_spect - targets_spect), 2), -1),-1) / input_sizes
            return sum(utt_loss)

        def SDR_loss(eps=1.0e-6):
            if self.crm:
                mask_r, mask_i = th.chunk(mask, 2, 2)
                estim_r = mixture_spect * th.cos(mixture_phase)
                estim_i = mixture_spect * th.sin(mixture_phase)
                x = self.inverse_stft(mask_r*estim_r - mask_i*estim_i, mask_r*estim_i + mask_i*estim_r, cplx=True)
            else:
                x = self.inverse_stft(mask*mixture_spect, mixture_phase, cplx=False)
            s = self.inverse_stft(targets_spect, targets_phase, cplx=False)
            x_zm = x - th.mean(x, dim=-1, keepdim=True)
            s_zm = s - th.mean(s, dim=-1, keepdim=True)
            if self.scale_inv: s_zm = th.clamp(th.sum(x_zm * s_zm, dim=-1, keepdim=True) / (l2norm(s_zm, keepdim=True)**2 + eps),min=1.0e-10) * s_zm
            utt_loss = - 20 * th.log10(eps + l2norm(s_zm) / (l2norm(x_zm - s_zm) + eps))
            return utt_loss
        
        utt_loss = MSE_loss() if self.loss == 'MSE' else SDR_loss()
        num_utts = input_sizes.shape[0]

        return th.sum(utt_loss) / num_utts
