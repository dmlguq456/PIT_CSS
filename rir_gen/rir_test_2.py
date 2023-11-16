import os

import librosa as audio_lib
import scipy.io.wavfile as wf
import numpy as np
import scipy.signal as ss
import random
import soundfile as sf
import colorednoise as cn
import math

MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps

h = np.load('./RIR_filter_v3/medium/target_RT_0.15.npy')

file = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tr/s1/01ac020r_0.96602_01jo030a_-0.96602.wav'
# file = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/cv/s1/011a010d_0.54422_20do010c_-0.54422.wav'
samps1, fs = audio_lib.load(file, sr=None)

file = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tr/s2/01ac020r_0.96602_01jo030a_-0.96602.wav'
# file = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/cv/s2/011a010d_0.54422_20do010c_-0.54422.wav'
samps2, fs = audio_lib.load(file, sr=None)

dataset = 'cv'
version = 'v3'
for beta in [0, 0.5, 1, 1.5, 2]:
    
    room = 'medium'
    rotate_num = 0
    RT = 0.15
    DIR = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/'+dataset+'/colorednoise_'+version
    path = DIR +'/noise_beta_'+str(beta)+'_room_'+room+'_'+'rotate_'+str(rotate_num)+'_RT_'+str(RT)
    file = path + '_sample0.wav'

    noise, fs = audio_lib.load(file, sr=None, mono=False)


    samps_tmp1 = ss.convolve(samps1[:,None], h[:,:,0,2], mode='full')
    samps_tmp1 = samps_tmp1[:samps1.shape[0],:]

    samps_tmp2 = ss.convolve(samps2[:,None], h[:,:,10,1], mode='full')
    samps_tmp2 = samps_tmp2[:samps2.shape[0],:]

    SNR1 = random.uniform(-2.5,2.5)
    SNR_factor1 = pow(10,-SNR1/20)
    SNR2 = random.uniform(-2.5,2.5)
    SNR_factor2 = pow(10,-SNR2/20)

    samps_rir = SNR_factor1*samps_tmp1 + SNR_factor2*samps_tmp2
    samps = SNR_factor1*samps1 + SNR_factor2*samps2
    sigma = np.std(samps)/np.std(samps_rir)
    samps_rir = sigma * samps_rir

    noise = np.transpose(noise[:,:samps2.shape[0]])
    SNR = 10
    sigma = np.std(samps_rir)/np.std(noise)
    samps_n = samps_rir + noise*sigma*pow(10,-SNR/20)


    file_rir = '/home/nas/user/Uihyeop/test_RIR_5/test_speech_RIR_mix_n_noisebeta'+str(beta)+'.wav'
    sf.write(file_rir, samps_n, fs)
