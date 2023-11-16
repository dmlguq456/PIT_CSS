import os

import librosa as audio_lib
import scipy.io.wavfile as wf
import numpy as np
import rir_generator as rir
import scipy.signal as ss
import random
import soundfile as sf
import math
from multiprocessing import Pool
import copy



def main(rt_list, rooms, mic_centers, mic_degs, random_seed):
    radius = 0.0425
    deg_60 = math.pi/3
    mic_height = 1
    for mic_center in mic_centers:
        for mic_deg_tmp in mic_degs:
            mic_deg = mic_deg_tmp*math.pi/180
            mic_array = [
                        [mic_center[0], mic_center[1], mic_height], 
                        [mic_center[0] + radius*math.cos(0*deg_60+mic_deg), mic_center[1] + radius*math.sin(0*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(1*deg_60+mic_deg), mic_center[1] + radius*math.sin(1*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(2*deg_60+mic_deg), mic_center[1] + radius*math.sin(2*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(3*deg_60+mic_deg), mic_center[1] + radius*math.sin(3*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(4*deg_60+mic_deg), mic_center[1] + radius*math.sin(4*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(5*deg_60+mic_deg), mic_center[1] + radius*math.sin(5*deg_60+mic_deg), mic_height]
                        ]
            for ch in range(1,7):
                for random_bias in random_seed:
                    mic_array_drop = copy.deepcopy(mic_array)
                    mic_array_drop[ch][0] += random_bias[0]
                    mic_array_drop[ch][1] += random_bias[1]
                    for key in rooms:
                        if key == 'small':
                            L=[4,3.5,3]
                            n_pos = [
                                    [0.1,0.1,0.1],[1,0.1,0.1],[2,0.1,0.1],[3,0.1,0.1], 
                                    [3.9,0.1,0.1],[3.9,1.1,0.1],[3.9,2.2,0.1],
                                    [3.9,3.4,0.1],[3,3.4,0.1],[2,3.4,0.1],[1,3.4,0.1],
                                    [0.1,3.4,0.1],[0.1,2.2,0.1],[0.1,1.1,0.1]
                                    ]
                        elif key == 'medium':
                            L=[5,4,6]
                            n_pos = [
                                    [0.1,0.1,0.1],[1,0.1,0.1],[2,0.1,0.1],[3,0.1,0.1],[4,0.1,0.1], 
                                    [4.9,0.1,0.1],[4.9,1,0.1],[4.9,2,0.1],[4.9,3,0.1],
                                    [4.9,3.9,0.1],[4,3.9,0.1],[3,3.9,0.1],[2,3.9,0.1],[1,3.9,0.1],
                                    [0.1,3.9,0.1],[0.1,3,0.1],[0.1,2,0.1],[0.1,1,0.1]
                                    ]

                        ## RIR generation
                        for rt60 in rt_list:
                            n_sample = int(rt60 * 16000)
                            h_n = np.zeros((n_sample,7,len(n_pos)))
                            for j in range(len(n_pos)):
                                h_n[:,:,j] = np.squeeze(rir.generate(
                                    c=340,
                                    fs=16000,
                                    r=mic_array_drop,
                                    s=n_pos[j],
                                    L=L,
                                    reverberation_time=rt60,
                                    nsample=n_sample,
                                ))
                            print("Generating RIR filters for RT = {} done.".format(rt60))
                            path = './RIR_filter_v7/'+key+'/mic_deg_'+str(mic_deg_tmp)+'_center_'+str(mic_center[0])+'_'+str(mic_center[1])+'_drop_ch_'+str(ch)+'_bias_'+str(random_bias[0])+'_'+str(random_bias[1])                    
                            os.makedirs(path, exist_ok=True)
                            np.save(path+'/noise_RT_'+str(rt60)+'.npy', h_n)

if __name__ == "__main__":
    rt_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    rooms = ['small', 'medium']
    # mic_centers = [[1.7,1.6],[1.8,1.55],[1.75,1.8],[1.9,1.95],[1.85,1.9]]
    random_seeds = [[-0.02, 0], [0.015, 0.015], [-0.02, 0.02]]
    mic_centers = [[1.8,1.8]]
    # mic_degs = [0, 10, 20, 30, 40, 50]
    mic_degs = [0]
    dict_list = [(rt_list,[room],[mic_center],[mic_deg],[random_seed]) for room in rooms for mic_center in mic_centers for mic_deg in mic_degs for random_seed in random_seeds]
    with Pool(len(dict_list)) as p:
        p.starmap(main, dict_list)
        
    