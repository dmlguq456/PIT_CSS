import os

import librosa as audio_lib
import scipy.io.wavfile as wf
import numpy as np
import rir_generator as rir
import scipy.signal as ss
import random
import soundfile as sf
import colorednoise as cn
import math


MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps


file = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tr/s1/01do030h_1.6205_024o030m_-1.6205.wav'
# file = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/cv/s1/011a010d_0.54422_20do010c_-0.54422.wav'
samps1, fs = audio_lib.load(file, sr=None)
file_s1 = '/home/nas/user/Uihyeop/test_RIR/test_speech_s1.wav'
sf.write(file_s1, samps1, fs)
sigma1 = pow(np.std(samps1),2) + pow(np.mean(samps1),2)

file = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/css/tr/s2/01do030h_1.6205_024o030m_-1.6205.wav'
# file = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/cv/s2/011a010d_0.54422_20do010c_-0.54422.wav'
samps2, fs = audio_lib.load(file, sr=None)
file_s2 = '/home/nas/user/Uihyeop/test_RIR/test_speech_s2.wav'
sf.write(file_s2, samps2, fs)
sigma2 = pow(np.std(samps2),2) + pow(np.mean(samps2),2)

# file = '/home/nas/DB/wsj0-mix/2speakers/wav16k/min/cv/mix/011a010d_0.54422_20do010c_-0.54422.wav'
# file = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/min/cv/mix_n/011a010d_0.54422_20do010c_-0.54422.wav'
# samps, fs = audio_lib.load(file, sr=None)
# samps_orig = samps
# sigma1 = np.std(samps)
# sigma1 = np.mean(np.abs(samps))
# noise = cn.powerlaw_psd_gaussian(1.5, samps.shape[0])
# sigma_n = np.std(noise)
# sigma_n = np.mean(np.abs(noise))
# SNR = 5
# samps = samps*pow(10,SNR/20) + noise*(sigma1/sigma_n)
# sigma2 = np.std(samps)
# sigma2 = np.mean(np.abs(samps))
# samps = samps*(sigma1/sigma2)
# sigma3 = np.std(samps)
# sigma3 = np.mean(np.abs(samps))
## RIR simulation for adding reverberation

# h = rir.generate(
#     c=340,
#     fs=fs,
#     r=[[1.5, 1.5, 2]],
#     s=[random.uniform(2,2.5),random.uniform(2,2.5),2],
#     L=[5,4,6],
#     reverberation_time=random.uniform(0.2, 0.4),
#     nsample=4096,
# )

h = np.zeros((4096,7,2))
radius = 0.045
deg_60 = math.pi/3
radius_spk = np.zeros((2,1))
deg_spk = np.zeros((2,1))
rt60 = random.uniform(0.2, 0.5)
for j in range(2):
    radius_spk[j] = random.uniform(0.5,1.5)
    if j == 0:
        deg_spk[j] = random.uniform(-math.pi,math.pi)
    else:
        if random.uniform(0,1) > 0.5:
            deg_spk[j] = deg_spk[0] + random.uniform(math.pi/6,math.pi)
        else:
            deg_spk[j] = deg_spk[0] - random.uniform(math.pi/6,math.pi)


    h[:,:,j] = np.squeeze(rir.generate(
        c=340,
        fs=16000,
        r=[
            [1.5, 1.5, 2], 
            [1.5 + radius*math.cos(0), 1.5 + radius*math.cos(0), 2],
            [1.5 + radius*math.cos(1*deg_60), 1.5 + radius*math.sin(1*deg_60), 2],
            [1.5 + radius*math.cos(2*deg_60), 1.5 + radius*math.sin(2*deg_60), 2],
            [1.5 + radius*math.cos(3*deg_60), 1.5 + radius*math.sin(3*deg_60), 2],
            [1.5 + radius*math.cos(4*deg_60), 1.5 + radius*math.sin(4*deg_60), 2],
            [1.5 + radius*math.cos(5*deg_60), 1.5 + radius*math.sin(5*deg_60), 2]
            ],
        s=[1.5 + radius_spk[j]*math.cos(deg_spk[j]), 1.5 + radius_spk[j]*math.sin(deg_spk[j]), 2],
        L=[5,4,6],
        reverberation_time=rt60,
        nsample=4096,
    ))

samps_tmp1 = ss.convolve(samps1[:,None], h[:,:,0], mode='full')
samps_tmp2 = ss.convolve(samps2[:,None], h[:,:,1], mode='full')

# n_tmp1 = ss.convolve(noise[:,0,None], h_n[:,:,0], mode='full')
# n_tmp2 = ss.convolve(noise[:,1,None], h_n[:,:,1], mode='full')
# n_tmp3 = ss.convolve(noise[:,2,None], h_n[:,:,2], mode='full')
# n_tmp4 = ss.convolve(noise[:,3,None], h_n[:,:,3], mode='full')
# noise = n_tmp1 + n_tmp2 + n_tmp3 + n_tmp4
# noise = noise[:samps2.shape[0],:]
# samps_tmp1=samps_tmp1*sigma1/(np.std(samps_tmp1)+ pow(np.mean(samps_tmp1),2))

# samps_tmp2=samps_tmp2*sigma2/(np.std(samps_tmp2) + pow(np.mean(samps_tmp2),2))


h_n = np.zeros((4096,7,18))
n_pos = [
        [0.1,0.1,6],[1,0.1,6],[2,0.1,6],[3,0.1,6],[4,0.1,6], 
        [4.9,0.1,6],[4.9,1,6],[4.9,2,6],[4.9,3,6],
        [4.9,3.9,6],[4,3.9,6],[3,3.9,6],[2,3.9,6],[1,3.9,6],
        [0.1,3.9,6],[0.1,3,6],[0.1,2,6],[0.1,1,6]
        ]
noise = np.zeros((samps1.shape[0],7))
noise_src = np.zeros((samps1.shape[0],18))
for j in range(18):
    h_n[:,:,j] = np.squeeze(rir.generate(
        c=340,
        fs=16000,
        r=[
            [1.5, 1.5, 2], 
            [1.5 + radius*math.cos(0), 1.5 + radius*math.cos(0), 2],
            [1.5 + radius*math.cos(1*deg_60), 1.5 + radius*math.sin(1*deg_60), 2],
            [1.5 + radius*math.cos(2*deg_60), 1.5 + radius*math.sin(2*deg_60), 2],
            [1.5 + radius*math.cos(3*deg_60), 1.5 + radius*math.sin(3*deg_60), 2],
            [1.5 + radius*math.cos(4*deg_60), 1.5 + radius*math.sin(4*deg_60), 2],
            [1.5 + radius*math.cos(5*deg_60), 1.5 + radius*math.sin(5*deg_60), 2]
            ],
        s=n_pos[j],
        L=[5,4,6],
        reverberation_time=rt60,
        nsample=4096,
    ))
    noise_src[:,j] = cn.powerlaw_psd_gaussian(1, samps1.shape[0])
    n_tmp = ss.convolve(noise_src[:,j,None], h_n[:,:,j], mode='full')
    noise += n_tmp[:samps2.shape[0],:]


samps = samps_tmp1[:samps1.shape[0],:] + samps_tmp2[:samps2.shape[0],:]

# for i in range(7):
    # noise[:,i] = cn.powerlaw_psd_gaussian(1.5, samps.shape[0])

SNR = random.uniform(5,15)
norm_tmp = (pow(np.std(samps),2) + pow(np.mean(samps),2))/(pow(np.std(noise),2) + pow(np.mean(noise),2))
samps_n = samps*pow(10,SNR/20) + noise*math.sqrt(norm_tmp)


# norm_tmp = math.sqrt(pow(np.std(samps_n),2) + pow(np.mean(samps_n),2))
norm_tmp = np.max(np.abs(samps_n))

file_rir = '/home/nas/user/Uihyeop/test_RIR/test_speech_RIR_s1.wav'
sf.write(file_rir, samps_tmp1/norm_tmp, fs)
file_rir = '/home/nas/user/Uihyeop/test_RIR/test_speech_RIR_s2.wav'
sf.write(file_rir, samps_tmp2/norm_tmp, fs)
file_rir = '/home/nas/user/Uihyeop/test_RIR/test_speech_RIR_mix.wav'
sf.write(file_rir, samps/norm_tmp, fs)
file_rir = '/home/nas/user/Uihyeop/test_RIR/test_speech_RIR_mix_n.wav'
sf.write(file_rir, samps_n/norm_tmp, fs)
