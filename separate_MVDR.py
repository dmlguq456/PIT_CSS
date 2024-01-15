"""
Using speaker mask produced by neural networks to separate single channel speech
"""

import argparse
from multiprocessing import dummy
import os
import pickle
import librosa as audio_lib

import numpy as np
import torch as th
import scipy.io as sio
from model.conformer import ConformerCSS
import soundfile as sf
import scipy.io.wavfile as wf

import scipy
from itertools import permutations
import copy
from MVDR import MVDR

from utils import stft, istft, parse_scps, parse_yaml, EPSILON


class Separator(object):
    def __init__(self, nnet, state_dict, cuda=False, gpu_id=0):
        if not os.path.exists(state_dict):
            raise RuntimeError(
                "Could not find state file {}".format(state_dict))
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        self.device = th.device("cuda:"+str(gpu_id) if cuda else "cpu")
        self.nnet = nnet.to(self.device)
        self.location = "cuda:"+str(gpu_id) if cuda else "cpu"
        # self.nnet.load_state_dict(th.load(state_dict, map_location=self.location))
        self.nnet.load_state_dict(th.load(state_dict, map_location=self.location)['model_state_dict'])
        # self.nnet.load_state_dict(
        #     th.load(state_dict, map_location=self.location))
        self.nnet.eval()
        self.mvdr = MVDR(ref_channel=0, diag_eps=1.0e-15, device=self.device)

    def seperate(self, spectra, mvn=None, apply_log=False, cos_sin_opt=False):
        """
            spectra: stft complex results T x F
            cmvn: python dict contains global mean/std
            apply_log: using log-spectrogram or not
        """
        if not np.iscomplexobj(spectra):
            raise ValueError("Input must be matrix in complex value")
        # compute (log)-magnitude spectrogram
        if len(spectra.shape) == 3:
            mix_feat = apply_cmvn(spectra[:,:,0]) if mvn else spectra[:,:,0]
            mix_feat = np.abs(mix_feat)                
            for i in range(1,spectra.shape[-1]):
                IPD = np.angle(spectra[:,:,i]) - np.angle(spectra[:,:,0])
                yr = np.cos(IPD)
                yi = np.sin(IPD)
                yrm = yr.mean(0, keepdims=True)
                yim = yi.mean(0, keepdims=True)
                if cos_sin_opt:
                    IPD = np.concatenate((yi - yim, yr - yrm), axis=-1)
                else:
                    IPD = np.arctan2(yi - yim, yr - yrm)
                mix_feat = np.concatenate((mix_feat, IPD), axis=-1)
            mix_feat = mix_feat.astype(np.float32)
        else:
            mix_feat = apply_cmvn(spectra) if mvn else spectra
            mix_feat = np.abs(mix_feat)                
            # if apply_log: mix_feat = np.log(np.maximum(mix_feat, EPSILON))

        with th.no_grad():
            out_masks = self.nnet(
                th.tensor(mix_feat, dtype=th.float32, device=self.device), train=False)
            # th.cuda.empty_cache() # personal edit
            # out_masks = self.nnet(
            #     th.tensor(mix_feat, dtype=th.float32, device=self.device),
            #     train=False)
        # spk_masks = [spk_mask.cpu().data.numpy() for spk_mask in out_masks]
        # spk_masks = [np.transpose(spk_mask.cpu().data.numpy()) for spk_mask in out_masks]
        spk_masks = [th.clamp(th.squeeze(th.transpose(spk_mask,0,-1)),min=EPSILON) for spk_mask in out_masks]
        # spk_masks = np.squeeze(spk_masks)
        # spk_masks = np.maximum(spk_masks, EPSILON)
        if len(spectra.shape) == 3:
            spk_mvdr = [th.transpose(self.mvdr(
                                    th.transpose(th.tensor(spectra),-1, 0).to(self.device),
                                    mask=th.transpose(th.tensor(spk_mask),-1,0),
                                    PF_beta=0.5
                                ),0,1).cpu().data.numpy()
                                for spk_mask in spk_masks]
            spk_masks=th.stack(spk_masks,dim=0).cpu().data.numpy()
            return spk_masks, [spectra[:,:,0] * spk_mask for spk_mask in spk_masks], spk_mvdr
        else:
            spk_masks=th.stack(spk_masks,dim=0).cpu().data.numpy()
            return spk_masks, [spectra * spk_mask for spk_mask in spk_masks] 
    
def apply_cmvn(feats):
    feats = feats - feats.mean(0, keepdims=True)
    feats = feats / feats.std(0, keepdims=True)
    return feats


def CSS_chunk_generator(CSS_config, stft_mat, frame_shift):
    chunk_len = int((CSS_config["N_h"] + CSS_config["N_c"] + CSS_config["N_f"]) * 16000 / frame_shift)
    chunk_shift = int(CSS_config["N_c"] * 16000 / frame_shift)
    N_h_frame = int(CSS_config["N_h"] * 16000 / frame_shift)
    N_f_frame = int(CSS_config["N_f"] * 16000 / frame_shift)
    total_frame = stft_mat.shape[0]
    init_zeros_paddings = np.zeros(
        (int(CSS_config["N_h"] * 16000 / frame_shift), 
            stft_mat.shape[1], 
            stft_mat.shape[2])
        )
    
    last_zeros_paddings = np.zeros(
        (chunk_shift - total_frame % chunk_shift + N_f_frame,
            stft_mat.shape[1], 
            stft_mat.shape[2])
        )

    stft_mat_pad = np.concatenate(
            (init_zeros_paddings, stft_mat, last_zeros_paddings),
            axis = 0)
    
    max_chunk_idx = (stft_mat_pad.shape[0] - N_h_frame) // chunk_shift
    chunk_list = [(chunk_shift*i, chunk_shift*i + chunk_len) for i in range(max_chunk_idx)]
    dummy_frame_len = last_zeros_paddings.shape[0] - N_f_frame
    return chunk_list, chunk_shift, N_h_frame, dummy_frame_len, stft_mat_pad

def masks_similarity(pre_mask, cur_mask, permute):
    corr = 0
    for s, t in enumerate(permute):
        corr += np.mean(pre_mask[s] * cur_mask[t])
    return corr, permute


def run(args):
    num_bins, config_dict = parse_yaml(args.config)
    dataloader_conf = config_dict["dataloader"]
    spectrogram_conf = config_dict["spectrogram_reader"]
    # Load cmvn
    # default: True
    IPD_sincos = config_dict["trainer"]["IPD_sincos"]
    config_dict["model"]["IPD_sincos"] = IPD_sincos
    apply_log = dataloader_conf["apply_log"] if "apply_log" in dataloader_conf else True
    if config_dict["model_type"] == "Conformer":
        nnet = ConformerCSS(**config_dict["model"])

    CSS_config = config_dict["inference"]["CSS_conf"]
    frame_length = spectrogram_conf["frame_length"]
    frame_shift = spectrogram_conf["frame_shift"]
    # window = spectrogram_conf["window"]
    window = scipy.signal.windows.hann(frame_length)
    if frame_length//4 == frame_shift:
        const = (2/3)**0.5
        window = const*window
    elif frame_length//2 == frame_shift:
        window = window**0.5
    
    separator = Separator(nnet, config_dict["inference"]["check_point_dir"], cuda=args.cuda, gpu_id=args.gpu_id)
    
    utt_dict = parse_scps(args.wave_scp)
    num_utts = 0
    for key, utt in utt_dict.items():
        try:
            if config_dict["model"]["num_mics"] == 1:
                samps_in,_ = audio_lib.load(utt, sr=None,mono=True)
            else:
                samps_in,_ = audio_lib.load(utt, sr=None,mono=False)
            samps_factor = 10
            samps_in = samps_in * samps_factor
            samps, stft_mat = stft(
                samps_in,
                frame_length=frame_length,
                frame_shift=frame_shift,
                window=window,
                center=True,
                return_samps=True)
        except FileNotFoundError:
            print("Skip utterance {}... not found".format(key))
            continue
        print("Processing utterance {}".format(key))
        num_utts += 1
        # norm = np.linalg.norm(samps, np.inf)
        mvn = config_dict["trainer"]["mvn"]
        if CSS_config["CSS"]:
            (chunk_list, chunk_shift, N_h, 
             dummy_frame_len, stft_mat_pad) = CSS_chunk_generator(CSS_config, stft_mat, frame_shift)
            spk_spect_chunk = [0,0]
            spk_mvdr_chunk = [0,0]
            for idx, (whole_begin, whole_end) in enumerate(chunk_list):
                stft_mat_chunk = stft_mat_pad[whole_begin:whole_end,:,:]
                spk_mask, spk_spect, spk_mvdr = separator.seperate(
                    stft_mat_chunk, mvn=mvn, apply_log=apply_log)
                if idx == 0:
                    pre_spk_mask_chunk = spk_mask[:,chunk_shift:]
                    spk_mask_chunk = copy.deepcopy(spk_mask[:,N_h:N_h+chunk_shift])
                    spk_mask_stack = copy.deepcopy(spk_mask)
                    mask_concat = copy.deepcopy(spk_mask_chunk)
                    mask_stack = copy.deepcopy(spk_mask_stack[:,:,:,None])
                    spk_spect_out = copy.deepcopy([spk[N_h:N_h+chunk_shift] for spk in spk_spect])
                    spk_mvdr_out = copy.deepcopy([spk[N_h:N_h+chunk_shift] for spk in spk_mvdr])
                else:
                    # Concatenate with the output order aligned
                    align_hypo = [masks_similarity(pre_spk_mask_chunk,spk_mask[:,:-chunk_shift,:],p)
                                    for p in permutations(range(config_dict["model"]["num_spks"]))]
                    align_hypo_val = np.array([i[0] for i in align_hypo])
                    mask_order = align_hypo[align_hypo_val.argmax()][1]
                    for s, t in enumerate(mask_order):
                        pre_spk_mask_chunk[s] = spk_mask[t,chunk_shift:]
                        spk_mask_chunk[s] = spk_mask[t,N_h:N_h+chunk_shift]
                        spk_mask_stack[s] = spk_mask[t]
                        spk_spect_chunk[s] = spk_spect[t][N_h:N_h+chunk_shift]
                        spk_mvdr_chunk[s] = spk_mvdr[t][N_h:N_h+chunk_shift]
                    mask_concat = np.concatenate((mask_concat,spk_mask_chunk),axis=1)
                    mask_stack = np.concatenate((mask_stack,spk_mask_stack[:,:,:,None]),axis=-1)
                    spk_spect_out = [np.concatenate((spk_spect_out[idx],spk),axis=0) for idx, spk in enumerate(spk_spect_chunk)]
                    spk_mvdr_out = [np.concatenate((spk_mvdr_out[idx],spk), axis=0) for idx, spk in enumerate(spk_mvdr_chunk)]
            spk_spect_out = [spk[:-dummy_frame_len] for spk in spk_spect_out]
            spk_mvdr_out = [spk[:-dummy_frame_len] for spk in spk_mvdr_out]
            mask_mean = mask_concat[:,:-dummy_frame_len,:].mean(-1)
            mask_mean = [np.convolve(mask_mean[i], np.ones(50)/50, mode='same') for i in range(2)]
            spk_mvdr_out = [spk_mvdr_out[i]*mask_mean[i][...,None] for i in range(2)]
        else:
            if len(stft_mat.shape) == 3:
                spk_mask, spk_spect_out, spk_mvdr_out = separator.seperate(
                    stft_mat, mvn=mvn, apply_log=apply_log, cos_sin_opt=IPD_sincos)
            else:
                spk_mask, spk_spect_out = separator.seperate(
                    stft_mat, mvn=mvn, apply_log=apply_log)
                spk_mvdr_out = None

        for index, stft_mat in enumerate(spk_spect_out):
            samps_out = istft(
                stft_mat,
                frame_length=frame_length,
                frame_shift=frame_shift,
                window=window,
                center=True,
                norm=config_dict["inference"]["save_conf"]["wav_norm"],
                fs=16000,
                nsamps=samps.shape[-1])
            file = os.path.join(args.dump_dir,'wav', '{}_{}.wav'.format(key[:-4], index))
            fdir = os.path.dirname(file)
            if fdir and not os.path.exists(fdir):
                os.makedirs(fdir)
            # sf.write(file, samps_out, 16000)
            wf.write(file, 16000, samps_out / samps_factor)
            # sf.write(file, samps_out / samps_factor, 16000)
            if spk_mvdr_out:
                samps_out = istft(spk_mvdr_out[index],
                    frame_length=frame_length,
                    frame_shift=frame_shift,
                    window=window,
                    center=True,
                    norm=config_dict["inference"]["save_conf"]["wav_norm"],
                    fs=16000,
                    nsamps=samps.shape[-1])
                file = os.path.join(args.dump_dir,'mvdr', '{}_{}.wav'.format(key[:-4], index))
                fdir = os.path.dirname(file)
                if fdir and not os.path.exists(fdir):
                    os.makedirs(fdir)
                # sf.write(file, samps_out, 16000)
                wf.write(file, 16000, samps_out / samps_factor)
                # sf.write(file, samps_out / samps_factor, 16000)
            if args.dump_mask:
                if CSS_config["CSS"]:
                    file_concat = os.path.join(args.dump_dir,'mask_concat','{}_{}.mat'.format(key[:-4], index))
                    fdir_concat = os.path.dirname(file_concat)
                    if fdir_concat and not os.path.exists(fdir_concat): os.makedirs(fdir_concat)
                    sio.savemat(file_concat, {"mask": mask_concat[index]})                    

                    file_stack = os.path.join(args.dump_dir,'mask_stack','{}_{}.mat'.format(key[:-4], index))
                    fdir_stack = os.path.dirname(file_stack)
                    if fdir_stack and not os.path.exists(fdir_stack): os.makedirs(fdir_stack)
                    sio.savemat(file_stack, {"mask": mask_stack[index]})
                    if index == 0:
                        file_stamp = os.path.join(args.dump_dir,'mask_stack','{}_stamp.mat'.format(key[:-4]))
                        fdir_stamp = os.path.dirname(file_stamp)
                        if fdir_stamp and not os.path.exists(fdir_stamp): os.makedirs(fdir_stamp)
                        sio.savemat(file_stamp, {"chunk_stamp": np.array(chunk_list)})
                else:
                    file = os.path.join(args.dump_dir,'mask','{}_{}.mat'.format(key[:-4], index))
                    fdir = os.path.dirname(file)
                    if fdir and not os.path.exists(fdir): os.makedirs(fdir)
                    sio.savemat(file, {"mask": spk_mask[index]})

    if CSS_config["CSS"]:
        print("Processed {} segments!".format(num_utts))
    else:
        print("Processed {} utterance!".format(num_utts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        "Command to seperate single-channel speech using masks generated by neural networks"
    )
    parser.add_argument(
        "--config", type=str, help="Location of training configure files")
    parser.add_argument(
        "--wave_scp",
        type=str,
        help="Location of input wave scripts in kaldi format")
    parser.add_argument(
        "--cuda",
        default=False,
        action="store_true",
        dest="cuda",
        help="If true, inference on GPUs")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="cache",
        dest="dump_dir",
        help="Location to dump seperated speakers")
    parser.add_argument(
        "--dump-mask",
        default=False,
        action="store_true",
        dest="dump_mask",
        help="If true, dump mask matrix")
    parser.add_argument(
        "--gpu-id",
        default=0)
    args = parser.parse_args()
    run(args)
