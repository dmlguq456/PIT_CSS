import os

import librosa as audio_lib
import scipy.io.wavfile as wf
import numpy as np
import rir_generator as rir
import scipy.signal as ss
import random
import soundfile as sf
import math
import argparse
from multiprocessing import Pool



def main(rt_list, rooms, mic_centers, mic_degs, radius, ref_mic_shift):
    ## RIR generation
    # radius = 0.0425
    # radius = 0.05
    deg_60 = math.pi/3
    deg_10 = math.pi/18
    radius_spk = [0.5, 0.75, 1, 1.25, 1.5]
    mic_height = 1
    for mic_center in mic_centers:
        for mic_deg_tmp in mic_degs:
            mic_deg = mic_deg_tmp*math.pi/180
            mic_array = [
                        [mic_center[0] + ref_mic_shift[0],                  mic_center[1] + ref_mic_shift[0],                  mic_height], 
                        [mic_center[0] + radius*math.cos(0*deg_60+mic_deg), mic_center[1] + radius*math.sin(0*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(1*deg_60+mic_deg), mic_center[1] + radius*math.sin(1*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(2*deg_60+mic_deg), mic_center[1] + radius*math.sin(2*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(3*deg_60+mic_deg), mic_center[1] + radius*math.sin(3*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(4*deg_60+mic_deg), mic_center[1] + radius*math.sin(4*deg_60+mic_deg), mic_height],
                        [mic_center[0] + radius*math.cos(5*deg_60+mic_deg), mic_center[1] + radius*math.sin(5*deg_60+mic_deg), mic_height]
                        ]
            # for rt60 in [0.5]:
            for key in rooms:
                if key == 'small':
                    L=[4,3.5,3]
                elif key == 'medium':
                    L=[5,4,6]
                for rt60 in rt_list:
                    n_sample = int(rt60 * 16000)
                    h = np.zeros((n_sample,7,36,len(radius_spk)))
                    os.makedirs('./RIR_filter_v10/'+key
                                +'/mic_deg_'+str(mic_deg_tmp)
                                +'_center_'+str(mic_center[0])+'_'+str(mic_center[1])
                                +'_mic_r_'+str(radius)
                                +'_ref_shift_'+str(ref_mic_shift[0])+'_'+str(ref_mic_shift[1]), 
                                exist_ok=True)
                    for i in range(36):
                        for j in range(len(radius_spk)): 
                            h[:,:,i,j] = np.squeeze(rir.generate(
                                c=340,
                                fs=16000,
                                r=mic_array,
                                s=[mic_center[0] + radius_spk[j]*math.cos(deg_10*i), mic_center[1] + radius_spk[j]*math.sin(deg_10*i), 2],
                                L=L,
                                reverberation_time=rt60,
                                nsample=n_sample
                            ))
                            # print("RT = {} : RIR filter for degree = {} and radius = {}  done.".format(rt60,10*i, radius_spk[j]))
                    print("Generating RIR filters for RT = {} done.".format(rt60))
                    
                    np.save('./RIR_filter_v10/'+key
                            +'/mic_deg_'+str(mic_deg_tmp)
                            +'_center_'+str(mic_center[0])+'_'+str(mic_center[1])
                            +'_mic_r_'+str(radius)
                            +'_ref_shift_'+str(ref_mic_shift[0])+'_'+str(ref_mic_shift[1])
                            +'/target_RT_'+str(rt60)+'.npy', 
                            h)

if __name__ == "__main__":
    rt_list = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    rooms = ['small', 'medium']
    mic_centers = [[1.8,1.8]]
    # mic_centers = [[1.7,1.6],[1.8,1.55],[1.75,1.8],[1.9,1.95],[1.85,1.9]]
    mic_degs = [0]
    ref_mic_shift_list = [[0,0], [0, 0.005], [0, -0.005],
                     [0.005,0],[0.005,0.005],[0.005, -0.005], 
                     [-0.005, 0], [-0.005, 0.005], [-0.005 -0.005], 
                     ]
    mic_radius_list = [0.0425]
    # mic_radius_list = [0.04, 0.0425, 0.045, 0.0475, 0.05]
    # mic_degs = [0, 10, 20, 30, 40, 50]
    dict_list = [(rt_list,[room],[mic_center],[mic_deg],mic_radius, ref_mic_shift) for room in rooms for mic_center in mic_centers for mic_deg in mic_degs for mic_radius in mic_radius_list for ref_mic_shift in ref_mic_shift_list]
    with Pool(len(dict_list)) as p:
        p.starmap(main, dict_list)