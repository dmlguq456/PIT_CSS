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
import math
from collections.abc import Sequence
from pesq import pesq_batch


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


class PITrainer_RI(object):
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
            # checkpoint_dict['model_state_dict'].pop('conformer.embed_inter1.0.bias')
            # checkpoint_dict['model_state_dict'].pop('conformer.embed_inter2.0.bias')
            self.nnet.load_state_dict(checkpoint_dict['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
            # checkpoint_dict['optimizer_state_dict'].pop('conformer.embed_inter1.0.bias')
            # checkpoint_dict['optimizer_state_dict'].pop('conformer.embed_inter2.0.bias')
            self.start_epoch = checkpoint_dict['epoch'] + 1
            # self.start_epoch = 1
        else:
            self.start_epoch = 1
        # self.optimizer.param_groups[0]["lr"] = 7.0e-5
        
        self.stft = STFT(stft_config["frame_length"],stft_config["frame_shift"],stft_config["window"], device=self.device)
        self.inverse_stft = iSTFT(stft_config["frame_length"], stft_config["frame_shift"], stft_config["window"], device=self.device)
        self.angle_feature_opt = angle_feature_opt
        if self.angle_feature_opt:
            self.ang_extractor = AngleFeature(
                num_bins=stft_config["frame_length"]/2+1,
                num_doas=self.num_spks,  # must known the DoA
                device=self.device
                )

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

        source_attr["spectrogram"], source_attr["phase"] = self.stft(mixture[:,:,0].to(self.device))
        mix_STFT = self.stft(tr(mixture,1,2).to(self.device))
        mix_STFT = tr(convert_complex(mix_STFT[0],mix_STFT[1]),1,3)
        for t in target:
            tmp = self.stft(t.to(self.device))
            target_attr["spectrogram"].append(tmp[0])
            target_attr["phase"].append(tmp[1])
        noise_attr["spectrogram"], noise_attr["phase"] = self.stft(noise[:,:,0].to(self.device))
        mix_feat = th.cat([th.real(mix_STFT), th.imag(mix_STFT)],dim=-1) # [ B, T, F, M*2]
            # mix_feat = th.abs(mix_feat)

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
            # masks = th.nn.parallel.data_parallel(self.nnet, (nnet_input, angle_diff), device_ids=self.gpuid)
            # cur_loss_s = self.PIT_loss(masks[:self.num_spks], input_sizes, source_attr, target_attr)
            cur_loss_s = self.PIT_loss_spec(masks[:self.num_spks], input_sizes, source_attr, target_attr)
            tot_loss_s += cur_loss_s.item() / self.num_spks
            # print(cur_loss_s.item() / self.num_spks)

            if len(masks) == self.num_spks:
                cur_loss = cur_loss_s / self.num_spks
            else:
                cur_loss_n = self.PIT_loss_spec(masks[self.num_spks], input_sizes, source_attr, noise_attr)
                # cur_loss_n = self.Noise_loss(masks[self.num_spks], input_sizes, source_attr, noise_attr)
                cur_loss = (cur_loss_s + cur_loss_n) / (self.num_spks + 1)            
                tot_loss_n += cur_loss_n.item()

            cur_loss.backward()
            if self.clip_norm:
                th.nn.utils.clip_grad_norm_(self.nnet.parameters(),
                                            self.clip_norm)
            self.optimizer.step()
            pbar.set_postfix({'Loss': tot_loss_s/num_batch, 'Loss_n': tot_loss_n/num_batch})
           # th.cuda.empty_cache() # personal edit
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
                input_sizes, source_attr, target_attr, noise_attr, nnet_input = extracted_feature
                # input_sizes, source_attr, target_attr, noise_attr, nnet_input, angle_diff, AF = extracted_feature

                num_batch += 1
                pbar.update(1)
                                
                masks = th.nn.parallel.data_parallel(self.nnet, nnet_input, device_ids=self.gpuid)
                cur_loss_s = self.PIT_loss_spec(masks[:self.num_spks], input_sizes, source_attr, target_attr)
                # cur_pesq_out = self.PIT_pesq(masks[:self.num_spks], input_sizes, source_attr, target_attr)
                # tot_pesq_out += cur_pesq_out.item()
                # cur_pesq_in = self.PIT_pesq(None, input_sizes, source_attr, target_attr)
                # tot_pesq_in += cur_pesq_in.item()
                tot_loss_s += cur_loss_s.item() / self.num_spks
                if len(masks) != self.num_spks:
                    cur_loss_n = self.PIT_loss_spec(masks[self.num_spks], input_sizes, source_attr, noise_attr)
                    tot_loss_n += cur_loss_n.item()

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
                # scheduler learning rate
                self.scheduler.step(valid_loss_src)
                logger.info(
                    "Loss(time/mini-batch) \n - Epoch {:2d}: train for source = {:.4f}({:.2f}s/{:d}) |"
                    " dev for source = {:.4f}({:.2f}s/{:d})".format(
                        epoch, train_loss_src, on_valid_start - on_train_start,
                        train_num_batch, valid_loss_src, on_valid_end - on_valid_start,
                        valid_num_batch))
                writer_src.add_scalars("Loss_src",{'train':train_loss_src, 'valid':valid_loss_src}, epoch)
                writer_src.flush()
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
                if epoch % 10 == 0:
                    on_test_start = time.time()
                    test_loss_src, test_num_batch = self.test(test_set)
                    on_test_end = time.time()
                    logger.info(
                        "Loss(time/mini-batch) \n - Epoch {:2d}: test for source = {:.4f} for freq and {:.4f} for time ({:.2f}s/{:d})"
                        .format(epoch, test_loss_src_freq, test_loss_src_time, on_test_start - on_test_end, test_num_batch)
                        )


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

        def MSE_loss(permute):
            loss_for_permute = []
            for s, t in enumerate(permute):
                if self.crm:
                    mask_r, mask_i = th.chunk(masks[s], 2, 2)
                    mixture_r, mixture_i = mixture_spect * th.cos(mixture_phase), mixture_spect * th.sin(mixture_phase)
                    sig_r, sig_i = targets_spect[t] * th.cos(targets_phase[t]), targets_spect[t] * th.sin(targets_phase[t])
                    loss_r =  th.sum(th.sum(th.pow(th.abs((mask_r*mixture_r - mask_i*mixture_i) - sig_r), 2), -1),-1) / input_sizes
                    loss_i =  th.sum(th.sum(th.pow(th.abs((mask_r*mixture_i + mask_i*mixture_r) - sig_i), 2), -1),-1) / input_sizes
                    utt_loss = loss_r + loss_i
                else:
                    utt_loss = th.sum(th.sum(th.pow(th.abs(masks[s]*mixture_spect - targets_spect[t]), 2), -1),-1) / input_sizes
                loss_for_permute.append(utt_loss)
            return sum(loss_for_permute)

        def A_SDR_loss(permute, eps=1.0e-6):
            loss_for_permute = []
            for s, t in enumerate(permute):
                if masks == None:
                    x = self.inverse_stft(mixture_spect, mixture_phase, cplx=False)
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
            return - 20 * th.log10((l2norm(s_zm_SA) + eps) / (l2norm(x_zm_SA - s_zm_SA) + tau*(l2norm(s_zm_SA) + eps) )) * self.num_spks


        pscore = th.stack( [A_SDR_loss(p) for p in permutations(range(self.num_spks))] )
        # pscore = th.stack( [(A_SDR_loss(p) if self.loss == 'A_SDR' else SA_SDR_loss(p) if self.loss == 'SA_SDR' else MSE_loss(p)) 
        #                     for p in permutations(range(self.num_spks))] )
        min_perutt, _ = th.min(pscore, dim=0)
        num_utts = input_sizes.shape[0]
        return th.sum(min_perutt) / num_utts


    def PIT_pesq(self, masks, input_sizes, source_attr, target_attr):

        input_sizes = input_sizes.to(self.device)
        mixture_spect = source_attr["spectrogram"].to(self.device)
        mixture_phase = source_attr["phase"].to(self.device)
        targets_spect = [t.to(self.device) for t in target_attr["spectrogram"]]
        targets_phase = [t.to(self.device) for t in target_attr["phase"]]
        if self.num_spks != len(targets_spect):
            raise ValueError(
                "Number targets do not match known speakers: {} vs {}".format(
                    self.num_spks, len(targets_spect)))


        def SA_SDR_loss(permute, eps=1.0e-6):
            s_zm_SA = x_zm_SA = th.tensor([]).to(self.device)
            for s, t in enumerate(permute):
                if self.crm:
                    mask_r, mask_i = th.chunk(masks[s], 2, 2)
                    estim_r, estim_i = mixture_spect * th.cos(mixture_phase), mixture_spect * th.sin(mixture_phase)
                    x = self.inverse_stft(mask_r*estim_r - mask_i*estim_i, mask_r*estim_i + mask_i*estim_r, cplx=True)
                else:
                    if masks==None:
                        x = self.inverse_stft(mixture_spect, mixture_phase, cplx=False)
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
            return th.tensor(pesq_batch(16000,s_zm_SA.cpu().data.numpy(),x_zm_SA.cpu().data.numpy(), mode='nb', n_processor=0))


        def A_SDR_loss(permute, eps=1.0e-6):
            loss_for_permute = []
            for s, t in enumerate(permute):
                x = self.inverse_stft(masks[s]*mixture_spect, mixture_phase, cplx=False)                    
                s = self.inverse_stft(targets_spect[t], targets_phase[t], cplx=False)

                x_zm = x - th.mean(x, dim=-1, keepdim=True)
                s_zm = s - th.mean(s, dim=-1, keepdim=True)
                if self.scale_inv: 
                    s_zm = th.sum(x_zm * s_zm, dim=-1, keepdim=True) / (l2norm(s_zm, keepdim=True)**2 + eps) * s_zm
                # utt_loss = - 20 * th.log10(eps + l2norm(s_zm) / (l2norm(x_zm - s_zm) + eps))
                utt_pesq = pesq_batch(16000,s.cpu().data.numpy(),x.cpu().data.numpy(), mode='nb', n_processor=0)
                loss_for_permute.append(utt_pesq)
            return sum(loss_for_permute)

        pscore = th.stack( [(SA_SDR_loss(p)) 
                            for p in permutations(range(self.num_spks))] )
        min_perutt, _ = th.max(pscore, dim=0)
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



class STFTBase(th.nn.Module):
    """
    Base layer for (i)STFT
    NOTE:
        1) Recommend sqrt_hann window with 2**N frame length, because it 
           could achieve perfect reconstruction after overlap-add
        2) Now haven't consider padding problems yet
    """
    def __init__(self,
                 frame_len,
                 frame_hop,
                 window="sqrt_hann",
                 device="cuda",
                 normalize=False):
        super(STFTBase, self).__init__()
        K = init_kernel(frame_len, frame_hop)
        self.K = th.nn.Parameter(K, requires_grad=False).to(device)
        self.stride = frame_hop
        self.window = window
        self.normalize = normalize
        self.num_bins = self.K.shape[0] // 2

    def extra_repr(self):
        return (f"window={self.window}, stride={self.stride}, " +
                f"kernel_size={self.K.shape[0]}x{self.K.shape[2]}, " +
                f"normalize={self.normalize}")
        

class iSTFT(STFTBase):
    """
    Inverse Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(iSTFT, self).__init__(*args, **kwargs)

    def forward(self, m, p, cplx=False, squeeze=False):
        """
        Accept phase & magnitude and output raw waveform
        args
            m, p: N x F x T
        return
            s: N x S
        """
        if p.dim() != m.dim() or p.dim() not in [2, 3]:
            raise RuntimeError("Expect 2D/3D tensor, but got {:d}D".format(
                p.dim()))
        # if F x T, reshape 1 x F x T
        if p.dim() == 2:
            p = th.unsqueeze(p, 0)
            m = th.unsqueeze(m, 0)
        if cplx:
            # N x 2F x T
            c = th.cat([m, p], dim=1)
        else:
            r = m * th.cos(p)
            i = m * th.sin(p)
            # N x 2F x T
            c = th.cat([r, i], dim=1)
        # N x 2F x T
        s = F.conv_transpose1d(c, self.K, stride=self.stride, padding=0)
        # N x S
        s = s.squeeze(1)
        if squeeze:
            s = th.squeeze(s)
        return s
    
class STFT(STFTBase):
    """
    Short-time Fourier Transform as a Layer
    """
    def __init__(self, *args, **kwargs):
        super(STFT, self).__init__(*args, **kwargs)

    def forward(self, x, cplx=False):
        """
        Accept (single or multiple channel) raw waveform and output magnitude and phase
        args
            x: input signal, N x C x S or N x S
        return
            m: magnitude, N x C x F x T or N x F x T
            p: phase, N x C x F x T or N x F x T
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} expect 2D/3D tensor, but got {:d}D signal".format(
                    self.__name__, x.dim()))
        # if N x S, reshape N x 1 x S
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
            # N x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x F x T
            r, i = th.chunk(c, 2, dim=1)
            # if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                # i = -i
        # else reshape NC x 1 x S
        else:
            N, C, S = x.shape
            x = x.reshape(N * C, 1, S)
            # x = x.view(N * C, 1, S)
            # NC x 2F x T
            c = F.conv1d(x, self.K, stride=self.stride, padding=0)
            # N x C x 2F x T
            c = c.reshape(N, C, -1, c.shape[-1])
            # c = c.view(N, C, -1, c.shape[-1])
            # N x C x F x T
            r, i = th.chunk(c, 2, dim=2)
            # if self.conjugate:
                # to match with science pipeline, we need to do conjugate
                # i = -i
        if cplx:
            return r, i
        m = (r**2 + i**2)**0.5
        p = th.atan2(i, r)
        return m, p

    

def init_kernel(frame_len, frame_hop):
    # FFT points
    N = frame_len
    # window
    W = th.hann_window(frame_len)
    if N//4 == frame_hop:
        const = (2/3)**0.5       
        W = const*W
    elif N//2 == frame_hop:
        W = W**0.5
    S = 0.5 * (N * N / frame_hop)**0.5

    # K = th.rfft(th.eye(N) / S, 1)[:frame_len]
    K = th.fft.rfft(th.eye(N) / S, dim=1)[:frame_len]
    K = th.stack((th.real(K), th.imag(K)),dim=2)
    # 2 x N/2+1 x F
    K = th.transpose(K, 0, 2) * W
    # N+2 x 1 x F
    K = th.reshape(K, (N + 2, 1, frame_len))
    return K


def convert_complex(m, p):
    r = m * th.cos(p)
    i = m * th.sin(p)
    # N x 2F x T
    
    c = th.complex(r, i)
    # N x 2F x T
    return c


def l2norm(mat, keepdim=False):
    return th.norm(mat, dim=-1, keepdim=keepdim)


def apply_cmvn(feats):
    feats = feats - feats.mean(1, keepdims=True)
    feats = feats / feats.std(1, keepdims=True)
    return feats


def IPD(spec_ref, spec, cos_sin_opt=False):
    ipd = th.angle(spec) - th.angle(spec_ref)
    yr = th.cos(ipd)
    yi = th.sin(ipd)
    yrm = yr.mean(1, keepdims=True)
    yim = yi.mean(1, keepdims=True)
    
    if cos_sin_opt:
        return th.cat([yi - yim, yr - yrm], dim=-1)
    else:
        return th.atan2(yi - yim, yr - yrm)
    
    
class AngleFeature(nn.Module):
    """
    Compute angle/directional feature
        1) num_doas == 1: we known the DoA of the target speaker
        2) num_doas != 1: we do not have that prior, so we sampled #num_doas DoAs 
                          and compute on each directions    
    """
    def __init__(self,
                 geometric="princeton",
                 sr=16000,
                 velocity=340,
                 num_bins=257,
                 num_doas=1,
                 device='cuda',
                 af_index="1,0;2,0;3,0;4,0;5,0;6,0"):
        super(AngleFeature, self).__init__()
        if geometric not in ["princeton"]:
            raise RuntimeError(
                "Unsupported array geometric: {}".format(geometric))
        self.geometric = geometric
        self.sr = sr
        self.num_bins = num_bins
        self.num_doas = num_doas
        self.velocity = velocity
        self.device = device
        split_index = lambda sstr: [
            tuple(map(int, p.split(","))) for p in sstr.split(";")
        ]
        # ipd index
        pair = split_index(af_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.af_index = af_index
        omega = th.tensor(
            [math.pi * sr * f / (num_bins - 1) for f in range(int(num_bins))]).to(self.device)
        # 1 x F
        self.omega = nn.Parameter(omega[None, :], requires_grad=False)

    def _oracle_phase_delay(self, doa):
        """
        Compute oracle phase delay given DoA
        args
            doa: N
        return
            phi: N x C x F or N x D x C x F
        """
        # device = doa.device
        # if self.num_doas != 1:
        #     # doa is a unused, fake parameter
        #     N = doa.shape[0]
        #     # N x D
        #     doa = th.linspace(0, math.pi * 2, self.num_doas + 1,
        #                       device=device)[:-1].repeat(N, 1)
        # for princeton
        # M = 7, R = 0.0425, treat M_0 as (0, 0)
        #      *3    *2
        #
        #   *4    *0    *1
        #
        #      *5    *6
        if self.geometric == "princeton":
            R = 0.0425
            zero = th.zeros_like(doa)
            # N x 7 or N x D x 7
            tau = R * th.stack([
                zero, -th.cos(doa), -th.cos(math.pi / 3 - doa),
                -th.cos(2 * math.pi / 3 - doa),
                th.cos(doa),
                th.cos(math.pi / 3 - doa),
                th.cos(2 * math.pi / 3 - doa)
            ],
                               dim=-1) / self.velocity
            # (Nx7x1) x (1xF) => Nx7xF or (NxDx7x1) x (1xF) => NxDx7xF
            phi = th.matmul(tau.unsqueeze(-1), -self.omega)
            return phi
        else:
            return None

    def extra_repr(self):
        return (
            f"geometric={self.geometric}, af_index={self.af_index}, " +
            f"sr={self.sr}, num_bins={self.num_bins}, velocity={self.velocity}, "
            + f"known_doa={self.num_doas == 1}")

    def _compute_af(self, ipd, doa):
        """
        Compute angle feature
        args
            ipd: N x C x F x T
            doa: DoA of the target speaker (if we known that), N 
                 or N x D (we do not known that, sampling D DoAs instead)
        return
            af: N x F x T or N x D x F x T
        """
        # N x C x F or N x D x C x F
        d = self._oracle_phase_delay(doa)
        d = d.unsqueeze(-1)
        if self.num_doas == 1:
            dif = d[:, self.index_l] - d[:, self.index_r]
            # N x C x F x T
            af = th.cos(ipd - dif)
            # on channel dimention (mean or sum)
            af = th.mean(af, dim=1)
        else:
            # N x D x C x F x 1
            dif = d[:, :, self.index_l] - d[:, :, self.index_r]
            # N x D x C x F x T
            af = th.cos(ipd.unsqueeze(1) - dif)
            # N x D x F x T
            af = th.mean(af, dim=2)
        
        return af

    def forward(self, p, doa, input_sizes):
        """
        Accept doa of the speaker & multi-channel phase, output angle feature
        args
            doa: DoA of target/each speaker, N or [N, ...]
            p: phase matrix, N x C x F x T
        return
            af: angle feature, N x F* x T or N x D x F x T (known_doa=False)
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError(
                "{} expect 3/4D tensor, but got {:d} instead".format(
                    self.__name__, p.dim()))
        # C x F x T => 1 x C x F x T
        if p.dim() == 3:
            p = p.unsqueeze(0)
        ipd = p[:, self.index_l] - p[:, self.index_r]

        if isinstance(doa, Sequence):
            if self.num_doas != 1:
                raise RuntimeError("known_doa=False, no need to pass "
                                   "doa as a Sequence object")
            # [N x F x T or N x D x F x T, ...]
            af = [self._compute_af(ipd, spk_doa) for spk_doa in doa]
            # N x F x T => N x F* x T
            af = th.cat(af, 1)
        else:
            # N x F x T or N x D x F x T
            af = self._compute_af(ipd, doa)

        input_sizes = input_sizes.int()
        for i in range(af.shape[0]):
            af[i,:,:,input_sizes[i]:] = 0
        return af

    
# class RIR_mixer(object):
    
#     def __init__(self, 
#                  gpuid,
#                  rir_dir,
#                  RT_list,
#                  num_mics,
#                  room_conf,
#                  num_spks=2
#                  ):
#         self.device = th.device('cuda:{}'.format(gpuid[0]))
#         self.RT_list = RT_list
#         self.rir_dir = rir_dir
#         self.num_mics = num_mics
#         self.rir_filter = {}
#         self.room_conf = room_conf
#         self.num_spks = num_spks
#         for room in room_conf:
#             self.rir_filter[room] = []
#             for RT in self.RT_list:
#                 self.rir_filter[room].append(th.tensor(np.load(self.rir_dir+'/'+room+'/target_RT_'+str(RT)+'.npy')).to(self.device))
            
#     def rir_filter_choice(self, num_utt, train=True):
#         ## Choose Room
#         # h_keys = list(self.rir_filter.keys())
#         room = random.choice(self.room_conf)
#         ## Choose Reverberation time
#         # RTs=[0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
#         RT = random.randint(0,len(self.RT_list)-1)            

#         ## Choose non-overlapped azimuth degrees for targets
#         num_deg = self.rir_filter[room][RT].shape[2]
#         batch_degs = [[] for _ in range(self.num_spks)]
#         for _ in range(num_utt):
#             degs = []
#             for i in range(self.num_spks):
#                 deg = random.randint(0,num_deg-1)
#                 while deg in degs: deg = random.randint(0,num_deg-1)
#                 degs.append(deg)
#                 batch_degs[i].append(deg)
            
#         batch_degs = [np.array(t) for t in batch_degs]        
#         ## Choose distances for targets
#         dist_len = self.rir_filter[room][RT].shape[3]
#         batch_dists = [[random.randint(0,dist_len-1) for i in range(num_utt)] for _ in range(self.num_spks)]
#         ## choose corresponding rir filter
#         batch_rotate = [random.randint(0,5) for _ in range(num_utt)]
#         h_target = []
#         for spk_idx in range(self.num_spks):
#             h_target.append([])
#             for utt_idx in range(num_utt):
#                 tmp = self.rir_filter[room][RT][:,:,batch_degs[spk_idx][utt_idx],batch_dists[spk_idx][utt_idx]]
#                 th.cat((tmp[:,[0]],th.roll(tmp[:,1:], batch_rotate[utt_idx],dims=1)),dim=1)
#                 h_target[spk_idx].append(tmp)
#             h_target[spk_idx]=th.stack(h_target[spk_idx],dim=0)
#         h_target = [th.tensor(h_target[spk_idx],dtype=th.float32) for spk_idx in range(self.num_spks)]

#         path_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/colorednoise_v2/' + ('tr' if train else 'cv')
#         # path_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/' + ('tr' if self.train else 'cv') + '/noise/pinknoise_room_' + room + '_RT_' + str(self.RT_list[RT])
#         beta = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
#         noises = []
#         # for utt_idx in range(num_utt):
#         file_n = ('noise_beta_' + str(random.choice(beta)) 
#                 + '_room_' + room 
#                 + '_RT_' + str(self.RT_list[RT]) 
#                 + '_sample' + str(random.randint(0,99 if train else 24)) 
#                 + '.wav')
#         noise, _ = audio_lib.load(path_n+'/'+file_n, sr=None, mono=False)
#         # noise = np.repeat(noise,num_utt,axis=0)
#         noise = th.tensor(noise, dtype=th.float32).to(self.device)
#         for utt_idx in range(num_utt):
#             noises.append(th.cat((noise[:,[0]],th.roll(noise[:,1:], batch_rotate[utt_idx],dims=1)),dim=1))
#         noises=th.stack(noises)

#         # h_target = [tr(th.tensor(h,dytpe=th.float32).to(self.device),0,1) for h in h_target]
#         return [tr(h,1,2) for h in h_target], noises
#         # return [tr(h,0,2) for h in h_target], tr(noises,1,2)

#     def RIR_mixing(self, srcs, num_utt, train=True):
#         # todo Now : h = N_mic X filter_length
#         h, noise = self.rir_filter_choice(num_utt, train)
#         samps_src = []
#         samps_src_rir = []
#         for idx, src in enumerate(srcs):
#             SNR = random.uniform(-2.5,2.5)
#             SNR_factor = pow(10,-SNR/20)
#             samps_tmp = src
#             samps_tmp = (samps_tmp * SNR_factor).to(self.device)
#             # samps_tmp = th.tensor(samps_tmp)
#             if samps_tmp.shape[0] > 256000: samps_tmp = samps_tmp[:256000]
#             h_src = th.flip(h[idx],[-1]) # torch conv is actually cross-correlation
#             if self.num_mics == 1:
#                 samps_tmp_rir = F.conv1d(samps_tmp, h_src[[0],:,:], groups=num_utt, padding='same') # samps_tmp = 1 X Batch X N_sample / h = N_mic X Batch X filter length
#                 samps_src_rir.append(samps_tmp_rir[:samps_tmp.shape[0],0])
#             else:
#                 samps_tmp_rir = F.conv1d(th.unsqueeze(samps_tmp,0), h_src.reshape(num_utt*self.num_mics,1,-1), padding='same',groups=num_utt)
#                 samps_src_rir.append(tr(samps_tmp_rir.reshape(num_utt,self.num_mics,-1),1,2))
#             samps_src.append(samps_tmp)
#         samps_rir_mix = sum(samps_src_rir)
#         sample_len = samps_rir_mix.shape[1]
#         samps_mix = sum(samps_src)

#         sigma = (th.std(samps_mix,dim=-1)/th.std(samps_rir_mix[:,:,0],dim=1)).unsqueeze(-1).unsqueeze(-1)
#         samps_rir_mix = samps_rir_mix*sigma
#         samps_src_rir = [src*sigma for src in samps_src_rir]
#         batch_SNR = th.tensor([random.uniform(5,25) for _ in range(num_utt)]).to(self.device)
#         norm_tmp = (th.std(samps_rir_mix[:,0,:],dim=-1)/th.std(noise[:,0,:],dim=-1)*pow(10,-batch_SNR/20)).unsqueeze(-1).unsqueeze(-1)
#         if self.num_mics == 1:
#             noise_SNR = tr(noise[:,0,:sample_len]*norm_tmp,1,2)
#         else:
#             noise_SNR = tr(noise[:,:,:sample_len]*norm_tmp,1,2)
#         samps_fin = samps_rir_mix + noise_SNR
#         return samps_fin, samps_src_rir, samps_src, noise_SNR
