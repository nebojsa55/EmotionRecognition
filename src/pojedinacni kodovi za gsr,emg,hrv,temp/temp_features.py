# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:36:24 2021

@author: HP
"""

import numpy as np
import pickle5 as pickle 
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt

temp_ch = 39

PATH = 'C:\\Users\\HP\\Desktop\\desktop\\master predmeti\\NI projekat\\data_preprocessed_python\\s01.dat'

with open(PATH, 'rb') as f:
    data = pickle.load(f, encoding = 'bytes') 
    
    
labels = pd.DataFrame(data[b'labels'], columns = ['valence','arousal', 'dominance', 'liking'])                                               

# data je dimenzija 40x40x8064 (num_videos x phy_channels x data)
data = data[b'data']
    
#%% OVO PROMENI

for i in tqdm.tqdm(range(40),desc = 'Number of trials'):
    
    signals = data[i,:,:] # signals from current trial
    temp = signals[temp_ch]
    

fs = 128
samples = 8064
time_axis = np.arange(0,samples/fs,1/fs);

plt.plot(time_axis, temp)

#%% feature-i su: mean, std, min, max, mean prvog izvoda, snaga u dva frekvencijska
    # opsega = 7 obelezja

features1 = np.array([np.mean(temp), np.std(temp), np.min(temp), np.max(temp),
                      np.mean(np.diff(temp))])

import pyphysio.indicators.FrequencyDomain as fq_ind
from pyphysio import EvenlySignal

temp_sig = EvenlySignal(values = temp,
                           sampling_freq = fs, 
                           signal_type = 'temp')

PSD1 = fq_ind.InBand(0,0.1,'welch')(temp_sig)
PSD2 = fq_ind.InBand(0.1,0.2,'welch')(temp_sig)

final_feat = np.append(features1, [np.mean(np.log(PSD1[1])), 
                                   np.mean(np.log(PSD2[1]))]) # 7 obelezja