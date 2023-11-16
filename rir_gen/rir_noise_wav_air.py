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
import librosa

from multiprocessing import Pool


MAX_INT16 = np.iinfo(np.int16).max
EPSILON = np.finfo(np.float32).eps
fs = 16000
# for dataset, nsample in {'tt':25}.items():

def f(h_n, sample, path, DIR_src, num_src):
    fs = 16000
    wav_len = fs*6
    noise = np.zeros((wav_len,7))
    noise_src = np.zeros((wav_len,num_src))
    noise_src = np.repeat(np.expand_dims(librosa.load(DIR_src+'/Noise_src_sample_'+str(sample)+'.wav')[0],axis=-1),num_src,axis=-1)
    # print(noise_src.shape)
    for j in range(num_src):
        # noise_src[:,j] = cn.powerlaw_psd_gaussian(wav_len)
        n_tmp = ss.convolve(noise_src[:,j,None], h_n[:,:,j], mode='full')
        noise += n_tmp[:wav_len,:]
    file = path+'/sample'+str(sample)+'.wav'
    sf.write(file, noise, fs)

# for dataset, nsample in {'tr':100}.items():
# mic_centers = [[1.7,1.6],[1.8,1.55],[1.75,1.8],[1.9,1.95],[1.85,1.9]]
# mic_degs = [0, 10, 20, 30, 40, 50]
mic_centers = [[1.8,1.8]]
mic_degs = [0]
for dataset, nsample in {'cv':120, 'tr':960, 'tt':120}.items():
    for mic_radius in [0.04, 0.0425, 0.045, 0.0475, 0.05]:
        for mic_center in mic_centers:
            for mic_deg_tmp in mic_degs:
                DIR = '/home/nas/user/Uihyeop/DB/Reverb_air_noise/diff_7ch/'+dataset
                DIR_src = '/home/nas/user/Uihyeop/DB/Reverb_air_noise/NOISE_seg_mono/'+dataset
                for room, num_src in {'medium':18, 'small':14}.items():
                    for RT in [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
                        print('Processing : noise generation for dataset = {}, Room = {}, RT = {}....'
                                .format(dataset, room, str(RT)))
                        path = DIR+'/noise_'+'room_'+room+'/mic_deg_'+str(mic_deg_tmp)+'_center_'+str(mic_center[0])+'_'+str(mic_center[1])+'_mic_r_'+str(mic_radius)+'_RT_'+str(RT)
                        os.makedirs(path,exist_ok=True)
                        h_n = np.load('./RIR_filter_v9/'+room+'/mic_deg_'+str(mic_deg_tmp)+'_center_'+str(mic_center[0])+'_'+str(mic_center[1])+'_mic_r_'+str(mic_radius)+'/noise_RT_'+str(RT)+'.npy')
                        dict_sample = [ (h_n, sample, path, DIR_src, num_src) for sample in range(nsample)]
                        # os.makedirs(path,exist_ok=True)
                        with Pool(50) as p:
                            p.starmap(f, dict_sample)
