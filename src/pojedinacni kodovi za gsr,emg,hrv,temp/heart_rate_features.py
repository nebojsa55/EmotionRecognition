# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 12:46:09 2021

@author: HP
"""

import numpy as np
import pickle5 as pickle 
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt

ppg_ch = 38

PATH = 'C:\\Users\\HP\\Desktop\\desktop\\master predmeti\\NI projekat\\data_preprocessed_python\\s01.dat'

with open(PATH, 'rb') as f:
    data = pickle.load(f, encoding = 'bytes') 
    
    
labels = pd.DataFrame(data[b'labels'], columns = ['valence','arousal', 'dominance', 'liking'])                                               

# data je dimenzija 40x40x8064 (num_videos x phy_channels x data)
data = data[b'data']
    
#%% OVO PROMENI

for i in tqdm.tqdm(range(40),desc = 'Number of trials'):
    
    signals = data[i,:,:] # signals from current trial
    ppg = signals[ppg_ch]

#%% HRV signal iz pletizmografije
fs = 128
samples = 8064
time_axis = np.arange(0,samples/fs,1/fs);   

#from scipy.signal import find_peaks
#peaks, _ = find_peaks(ppg, distance = 60) # distanca na osnovu cinjenice da RR
#                                        # interval traje 0.6-1.2 s
#
##plt.plot(time_axis,ppg)
##plt.plot(peaks, ppg[peaks], 'rx', markersize = 7, mew = 2)
                                        
import pyphysio.estimators.Estimators as estimators
from pyphysio import EvenlySignal

ppg_sig = EvenlySignal(values = ppg,
                           sampling_freq = fs, 
                           signal_type = 'PPG')

HRV = estimators.BeatFromBP()(ppg_sig) #intervali u sekundama

#%% feature-i iz elderly2019 najvece stat znacajnosti - first derivative mean,
    # RMS, arc length, area perimeter ratio = 4

arc_length = 0
for i in range(len(HRV)-1):
    arc_length += np.sqrt(1 + np.square(HRV[i + 1] - HRV[i]))

arc_length /= len(HRV)

area_per_ratio = np.sum(HRV)/len(HRV)/arc_length

features1 = np.array([np.mean(np.diff(HRV)), np.sqrt(np.mean(np.square(HRV))),
                      arc_length, area_per_ratio])

#%% feature-i iz DEAP rada - mean i std HRV, snage 3 frekv opsega i odnos snaga
    # dva opsega = 6
    
import pyphysio.indicators.FrequencyDomain as fq_ind

HRV_sig = EvenlySignal(values = HRV,
                           sampling_freq = len(HRV)/60, 
                           signal_type = 'HRV')

low_PSD = fq_ind.InBand(0.01,0.08,'welch')(HRV_sig)
med_PSD = fq_ind.InBand(0.08,0.15,'welch')(HRV_sig)
high_PSD = fq_ind.InBand(0.15,0.5,'welch')(HRV_sig)

range1 = fq_ind.InBand(0.04,0.15,'welch')(HRV_sig)
range2 = fq_ind.InBand(0.15,0.5,'welch')(HRV_sig)

features2 = np.array([np.mean(HRV), np.std(HRV), np.mean(np.log(low_PSD[1])), 
                      np.mean(np.log(med_PSD[1])), np.mean(np.log(high_PSD[1])),
                      np.mean(range1[1])/np.mean(range2[1])])
    
final_feat = np.append(features1,features2) # 10 obelezja