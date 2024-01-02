import os
import random
import pickle

import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch as th
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence, pad_sequence
from torch.nn.functional import conv1d
from utils import parse_scps, stft, apply_cmvn, EPSILON, get_logger, IPD
import scipy.signal as ss
import scipy.io.wavfile as scipy_audio
import librosa as audio_lib
import colorednoise as cn
import math
import time
import colorednoise as cn

logger = get_logger(__name__)


class SpectrogramReader(object):
    """
        Wrapper for short-time fourier transform of dataset
    """

    def __init__(self, wave_scp, rir_list, noise_dir, rir_mode, train, num_mics, max_frame, loss, **kwargs):
        self.num_mics = num_mics
        self.stft_kwargs = kwargs
        self.max_len = (1+max_frame)*self.stft_kwargs["frame_shift"]
        self.train = train
        self.rir_filter = rir_list[0]
        self.rir_mode = rir_mode
        if self.rir_mode > 0:
            """
            RIR mode = 0 : random
            RIR mode = 1 : fixed + 0~20
            RIR mode = 2 : fixed + 30~90
            RIR mode = 3 : fixed + 100~180           
            """
            random.seed(0)
        self.rir_range = (1,18) if rir_mode == 0 else (0,2) if rir_mode == 1 else (3,9) if rir_mode == 2 else (10,18)
        self.RT_list = rir_list[1]
        self.noise_dir = noise_dir
        self.loss = loss
        for wave_scp_src in wave_scp:
            if not os.path.exists(wave_scp_src):
                raise FileNotFoundError("Could not find file {}".format(wave_scp_src))
        self.wave_dict = [parse_scps(wave_scp_src) for wave_scp_src in wave_scp]
        self.wave_keys = [key for key in self.wave_dict[0].keys()]
        logger.info(
            "Create SpectrogramReader for {} with {} utterances".format(
                wave_scp, len(self.wave_dict)))

    def __len__(self):
        return len(self.wave_dict[0])

    def __contains__(self, key):
        return key in self.wave_dict

    # random pick of RIR generation filter
    def rir_filter_choice(self):
        ## Choose Room
        h_keys = list(self.rir_filter.keys())
        room = random.choice(h_keys)

        #! RIR_filter_v9
        mic_deg = 0
        radius_set = [0.04, 0.0425, 0.045]
        idx2 = random.randint(0,len(radius_set)-1)
        # radius_set = [0.04, 0.0425, 0.045, 0.0475, 0.05]
        radius = radius_set[idx2]
        RT = random.randint(0,len(self.RT_list)-1)
        num_deg, dist_len = self.rir_filter[room][idx2][RT].shape[2:]

        # dist_len = self.rir_filter[room][idx2][RT].shape[3]
        ## Choose non-overlapped azimuth degrees for targets
        unit_deg = (360/num_deg)*(math.pi/180)
        degs = []
        degs_diff = []
        # for i in range(len(self.wave_dict)):
        #     deg = random.randint(0,num_deg-1)
        #     while deg in degs:
        #         deg = random.randint(0,num_deg-1)
        #     degs.append(deg)
        #     if i > 0: degs_diff.append(np.mod(abs(degs[0]-deg)*unit_deg, math.pi) )
        deg = random.randint(0,num_deg-1)
        degs.append(deg)
        for i in range(len(self.wave_dict)-1):
            if self.rir_mode == 0:
                while deg in degs:
                    deg_delta = random.randint(self.rir_range[0],self.rir_range[1])
                    deg = (degs[0] + deg_delta)%num_deg if random.random() > 0.5 else degs[0] - deg_delta
            else:
                deg_delta = random.randint(self.rir_range[0],self.rir_range[1])
                deg = (degs[0] + deg_delta)%num_deg if random.random() > 0.5 else degs[0] - deg_delta
            degs.append(deg)


        ## Choose distances for targets
        dists = [random.randint(0,dist_len-1) for i in range(len(self.wave_dict))]
        #! RIR_filter_v9
        
        h_target = [self.rir_filter[room][idx2][RT][:,:,degs[idx],dists[idx]] for idx in range(2)]

        # rotate_num = random.randint(0,6)
        # h_target = [np.roll(h, rotate_num, axis=1) for h in h_target]
        ## choose noise sample with same room and RT configuration
        # path_n = '/data/Uihyeop/colorednoise_v2/' + ('tr' if self.train else 'cv')
        path_n = self.noise_dir + ('tr' if self.train else 'cv')
        # path_n = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/colorednoise_v4/' + ('tr' if self.train else 'cv')
        beta = [0.5, 0.75, 1, 1.25, 1.5]
        centers = [[1.7, 1.6], [1.8, 1.55], [1.9, 1.95], [1.75, 1.8], [1.85, 1.9]]
        file_n = ('noise_beta_' + str(random.choice(beta)) 
                  + '_room_' + room 
                  + '/mic_deg_' + str(mic_deg)
                #! RIR_filter_v9
                  + '_center_1.8_1.8_mic_r_' + str(radius)
                  
                  + '_RT_' + str(self.RT_list[RT]) 
                  + '/sample' + str(random.randint(0,39 if self.train else 9)) 
                  + '.wav')
        # file_n = 'sample_' + str(random.randint(0, 119 if self.train else 14)) + '.wav'
        noise, _ = audio_lib.load(path_n+'/'+file_n, sr=None, mono=False)
        # noise = np.append(noise[[0],:],np.roll(noise[1:,:], rotate_num,axis=0),axis=0)
        # noise = np.repeat(noise[:,5000:-5000], 5, axis=1)
        # degs = [np.mod(deg*unit_deg + rotate_num*math.pi/3, 2*math.pi) for deg in degs]
        return h_target, np.transpose(noise)
        
    def RIR_mixing(self, key):
        h, noise = self.rir_filter_choice()
        h_d = [np.zeros(h_tmp.shape[0]) for h_tmp in h]
        h_d_2 = [np.zeros(h_tmp.shape[0]) for h_tmp in h]
        for i in range(len(h)):
            idx = h[i][:,0].argmax()
            if self.loss == "MSE":
                h_d[i][0:idx+64] = h[i][0:idx+64,0]
            else:
                # h_d[i][0:idx+1024] = h[i][0:idx+1024,0]
                h_d[i][0:idx+32] = h[i][0:idx+32,0]
                h_d_2[i][0:idx+32] = h[i][0:idx+32,0]
                # h_d[i][0:idx+512] = h[i][0:idx+512,0]
                # h_d_2[i][0:idx+512] = h[i][0:idx+512,0]

        samps_src = []
        samps_src_rir = []
        samps_src_rir_d = []
        samps_src_rir_d_2 = []
        files = [wave_dict_src[key] for wave_dict_src in self.wave_dict]
        for idx, file in enumerate(files):
            SNR = random.uniform(-2.5,2.5)
            SNR_factor = pow(10,-SNR/20)
            if not os.path.exists(file):
                raise FileNotFoundError("Input file {} do not exists!".format(file))
            samps_tmp, _ = audio_lib.load(file, sr=None)
            samps_tmp = samps_tmp * SNR_factor
            if samps_tmp.shape[0] > self.max_len:
                samps_tmp = samps_tmp[:self.max_len]
            if self.num_mics == 1:
                samps_tmp_rir = ss.convolve(samps_tmp[:,None], h[idx][:,0].reshape(-1,1), mode='full')
                samps_src_rir.append(samps_tmp_rir[:samps_tmp.shape[0],0])
            elif self.num_mics == 3:
                samps_tmp_rir = ss.convolve(samps_tmp[:,None], h[idx][:,[0,1,4]], mode='full')
                samps_src_rir.append(samps_tmp_rir[:samps_tmp.shape[0],:])                
            else:
                samps_tmp_rir = ss.convolve(samps_tmp[:,None], h[idx], mode='full')
                samps_src_rir.append(samps_tmp_rir[:samps_tmp.shape[0],:])

            samps_tmp_rir_d = ss.convolve(samps_tmp[:,None], h_d[idx].reshape(-1,1), mode='full')
            samps_src_rir_d.append(samps_tmp_rir_d[:samps_tmp.shape[0],0])
            samps_tmp_rir_d_2 = ss.convolve(samps_tmp[:,None], h_d_2[idx].reshape(-1,1), mode='full')
            samps_src_rir_d_2.append(samps_tmp_rir_d_2[:samps_tmp.shape[0],0])
            samps_src.append(samps_tmp)

        samps_mix = sum(samps_src)
        samps_rir_mix = sum(samps_src_rir)
        sigma = (np.std(samps_mix) + 1.0e-6)/(np.std(samps_rir_mix) + 1.0e-6)
        # # ! amplification of input mixture to be masked
        samps_rir_mix = samps_rir_mix*sigma
        samps_src_rir = [src*sigma for src in samps_src_rir]
        sigam_d = [0.9*(np.std(src) + 1.0e-6)/(np.std(samps_src_rir_d[idx]) + 1.0e-6) for idx, src in enumerate(samps_src_rir)]
        sigam_d_2 = [0.9*(np.std(src) + 1.0e-6)/(np.std(samps_src_rir_d_2[idx]) + 1.0e-6) for idx, src in enumerate(samps_src_rir)]
        samps_src_rir_d = [src*sigam_d[idx] for idx, src in enumerate(samps_src_rir_d)]
        samps_src_rir_d_2 = [src*sigam_d_2[idx] for idx, src in enumerate(samps_src_rir_d_2)]
        # samps_src_rir_d = [src*sigma*(np.std(h[idx])/np.std(h_d[idx])) for idx, src in enumerate(samps_src_rir_d)]
        # samps_src_rir_d_2 = [src*sigma*(np.std(h[idx])/np.std(h_d[idx])) for idx, src in enumerate(samps_src_rir_d_2)]
        SNR = random.uniform(0,20)
        norm_tmp = (np.std(samps_rir_mix) + 1.0e-6)/(np.std(noise) + 1.0e-6)
        if self.num_mics == 1:
            noise_SNR = noise[:samps_rir_mix.shape[0],0]*norm_tmp*pow(10,-SNR/20)
        elif self.num_mics == 3:
            noise_SNR = noise[:samps_rir_mix.shape[0],[0,1,4]]*norm_tmp*pow(10,-SNR/20)            
        else:
            noise_SNR = noise[:samps_rir_mix.shape[0],:]*norm_tmp*pow(10,-SNR/20)
        samps_fin = samps_rir_mix + noise_SNR

        # return samps_mix, samps_src, samps_src, samps_src[0]
        return samps_fin, samps_src_rir_d_2, samps_src_rir_d, noise_SNR

    '''
    # sequential index
    def __iter__(self):d
        for key in self.wave_dict:
            yield key, self._load(key)
    '''
    # random index

    def __getitem__(self, key):
        for i in range(len(self.wave_dict)):
            if key not in self.wave_dict[i]:
                raise KeyError("Could not find utterance {}".format(key))
        return self.RIR_mixing(key)


class Datasets(object):
    def __init__(self, mix_reader):
        self.mix_reader = mix_reader
        self.key_list = mix_reader.wave_keys

    def __len__(self):
        return len(self.mix_reader)

    def __getitem__(self, index):
        key = self.key_list[index]
        mix, ref_rir, ref, noise = self.mix_reader[key]
        
        return {"num_sample" : mix.shape[0], "mix":mix, "ref_rir":ref_rir, "ref":ref, "noise":noise}
