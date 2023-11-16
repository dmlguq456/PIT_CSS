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


def main(rt_list, rooms, mic_centers, mic_degs, radius):
    # radius = 0.05
    # radius = 0.0425
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
                    os.makedirs('./RIR_filter_v9/'+key+'/mic_deg_'+str(mic_deg_tmp)+'_center_'+str(mic_center[0])+'_'+str(mic_center[1])+'_mic_r_'+str(radius), exist_ok=True)
                    for j in range(len(n_pos)):
                        h_n[:,:,j] = np.squeeze(rir.generate(
                            c=340,
                            fs=16000,
                            r=mic_array,
                            s=n_pos[j],
                            L=L,
                            reverberation_time=rt60,
                            nsample=n_sample,
                        ))
                    print("Generating RIR filters for RT = {} done.".format(rt60))                    
                    np.save('./RIR_filter_v9/'+key+'/mic_deg_'+str(mic_deg_tmp)+'_center_'+str(mic_center[0])+'_'+str(mic_center[1])+'_mic_r_'+str(radius)+'/noise_RT_'+str(rt60)+'.npy', h_n)

if __name__ == "__main__":
    rt_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    rooms = ['small', 'medium']
    mic_centers = [[1.8,1.8]]
    # mic_centers = [[1.7,1.6],[1.8,1.55],[1.75,1.8],[1.9,1.95],[1.85,1.9]]
    mic_degs = [0]
    # mic_degs = [0, 10, 20, 30, 40, 50]
    mic_radius_list = [0.04, 0.0425, 0.045, 0.0475, 0.05]
    dict_list = [(rt_list,[room],[mic_center],[mic_deg],mic_radius) for room in rooms for mic_center in mic_centers for mic_deg in mic_degs for mic_radius in mic_radius_list]
    with Pool(len(dict_list)) as p:
        p.starmap(main, dict_list)
        
    