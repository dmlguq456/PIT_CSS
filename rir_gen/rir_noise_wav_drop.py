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

from multiprocessing import Pool


MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps
version = 'v7'
fs = 16000
# for dataset, nsample in {'tt':25}.items():

def f(h_n, sample, path, beta, num_src):
    fs = 16000
    wav_len = fs*18
    noise = np.zeros((wav_len,7))
    noise_src = np.zeros((wav_len,num_src))
    for j in range(num_src):
        noise_src[:,j] = cn.powerlaw_psd_gaussian(beta, wav_len)
        n_tmp = ss.convolve(noise_src[:,j,None], h_n[:,:,j], mode='full')
        noise += n_tmp[:wav_len,:]
    file = path+'/sample'+str(sample)+'.wav'
    sf.write(file, noise, fs)

# for dataset, nsample in {'tr':100}.items():
# mic_centers = [[1.7,1.6],[1.8,1.55],[1.75,1.8],[1.9,1.95],[1.85,1.9]]
# mic_degs = [0, 10, 20, 30, 40, 50]
mic_degs = [0]
mic_centers = [[1.8,1.8]]

# for dataset, nsample in {'tr':40}.items():
# for dataset, nsample in {'cv':10}.items():
for dataset, nsample in {'tt':10}.items():
# for dataset, nsample in {'cv':10, 'tr':40, 'tt':10}.items():
    for ch in range(1,7):
        for random_bias in [[-0.02, 0], [0.015, 0.015], [-0.02, 0.02]]:
            for mic_center in mic_centers:
                for mic_deg_tmp in mic_degs:
                    DIR = '/home/nas/user/Uihyeop/DB/wsj0-mix/2speakers/wav16k/colorednoise_'+version+'/'+dataset
                    # for room, num_src in {'big':28}.items():
                    for room, num_src in {'medium':18, 'small':14}.items():
                        for RT in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
                            for beta in [0.5, 0.75, 1, 1.25, 1.5]:
                            # for beta in [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]:
                                print('Processing : noise generation for dataset = {}, Room = {}, RT = {}...., beta = {}'
                                        .format(dataset, room, str(RT), str(beta)))
                                path_common = room+'/mic_deg_'+str(mic_deg_tmp)+'_center_'+str(mic_center[0])+'_'+str(mic_center[1])+'_drop_ch_'+str(ch)+'_bias_'+str(random_bias[0])+'_'+str(random_bias[1])
                                path = DIR+'/noise_beta_'+str(beta)+'_room_'+path_common+'_'+str(mic_center[1])+'_RT_'+str(RT)
                                os.makedirs(path,exist_ok=True)
                                h_n = np.load('./RIR_filter_v7/'+path_common+'/noise_RT_'+str(RT)+'.npy')
                                dict_sample = [ (h_n, sample, path, beta, num_src) for sample in range(nsample)]
                                # os.makedirs(path,exist_ok=True)
                                with Pool(50) as p:
                                    p.starmap(f, dict_sample)
