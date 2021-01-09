# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 12:46:53 2021

@author: HP
"""

import numpy as np
import pickle5 as pickle 
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt

zemg_ch = 34
temg_ch = 35

PATH = 'C:\\Users\\HP\\Desktop\\desktop\\master predmeti\\NI projekat\\data_preprocessed_python\\s01.dat'

with open(PATH, 'rb') as f:
    data = pickle.load(f, encoding = 'bytes') 
    
    
labels = pd.DataFrame(data[b'labels'], columns = ['valence','arousal', 'dominance', 'liking'])                                               

# data je dimenzija 40x40x8064 (num_videos x phy_channels x data)
data = data[b'data']
    
#%% OVO PROMENI

for i in tqdm.tqdm(range(40),desc = 'Number of trials'):
    
    signals = data[i,:,:] # signals from current trial
    zemg = signals[zemg_ch]
    temg = signals[temg_ch]
    

fs = 128
samples = 8064
time_axis = np.arange(0,samples/fs,1/fs);

#plt.plot(time_axis, zemg)
#plt.figure()
#plt.plot(time_axis, temg)

#%% feature-i iz DEAP - snaga u opsegu 4-40hz = 2 obelezja

from pyphysio import EvenlySignal
import pyphysio.indicators.FrequencyDomain as fq_ind

zemg_sig = EvenlySignal(values = zemg,
                           sampling_freq = fs, 
                           signal_type = 'zEMG')
temg_sig = EvenlySignal(values = temg,
                           sampling_freq = fs, 
                           signal_type = 'tEMG')

zemg_PSD = fq_ind.InBand(4,40,'welch')(zemg_sig)
temg_PSD = fq_ind.InBand(4,40,'welch')(temg_sig)

#%% uklanjanje drifta konvolucijom

w = 256
mask = np.ones((1,w))/w
mask = mask[0,:]

convolved_zemg = np.convolve(zemg, mask, 'same')
convolved_temg = np.convolve(temg, mask, 'same')

zemg_dc = zemg - convolved_zemg
temg_dc = temg - convolved_temg

#%% izvlace se mean, std, mean prvog i drugog izvoda = 8 (iz drugog rada)

features1z = np.array([np.mean(zemg_dc), np.std(zemg_dc), np.mean(np.diff(zemg_dc)), 
                      np.mean(np.diff(np.diff(zemg_dc)))])
features1t = np.array([np.mean(temg_dc), np.std(temg_dc), np.mean(np.diff(temg_dc)), 
                      np.mean(np.diff(np.diff(temg_dc)))])

#%% feature-i iz LP gsr signala (0.3 Hz)
# izvlace se mean, std, mean prvog i drugog izvoda = 8
import pyphysio.filters.Filters as filters

zemg_dc = EvenlySignal(values = zemg_dc,
                           sampling_freq = fs, 
                           signal_type = 'zEMG')
temg_dc = EvenlySignal(values = temg_dc,
                           sampling_freq = fs, 
                           signal_type = 'tEMG')

zemg_lp = filters.FIRFilter([0.01,0.3],[0,0.32])(zemg_dc)
temg_lp = filters.FIRFilter([0.01,0.3],[0,0.32])(temg_dc)

features2z = np.array([np.mean(np.log(zemg_PSD[1])), np.mean(zemg_lp), 
                       np.std(zemg_lp), np.mean(np.diff(zemg_lp)), 
                      np.mean(np.diff(np.diff(zemg_lp)))])
features2t = np.array([np.mean(np.log(temg_PSD[1])), np.mean(temg_lp), 
                       np.std(temg_lp), np.mean(np.diff(temg_lp)), 
                      np.mean(np.diff(np.diff(temg_lp)))])

#%% feature-i su: broj myoresponses u lp i vlp signalu, i njihov odnos = 6
    
zemg_vlp = filters.FIRFilter([0.01,0.08],[0,0.1])(zemg_dc)
temg_vlp = filters.FIRFilter([0.01,0.08],[0,0.1])(temg_dc) 
    
from scipy.signal import find_peaks

peaksz_lp, _ = find_peaks(zemg_lp, distance = 128)
peaksz_vlp, _ = find_peaks(zemg_vlp, distance = 128)

peakst_lp, _ = find_peaks(temg_lp, distance = 128)
peakst_vlp, _ = find_peaks(temg_vlp, distance = 128)

#plt.plot(temg_vlp)
#plt.plot(peakst_vlp, temg_vlp[peakst_vlp], 'rx', markersize = 7, mew = 2)

features3z = np.array([np.size(peaksz_lp), np.size(peaksz_vlp),
                     np.size(peaksz_lp)/np.size(peaksz_vlp)])
features3t = np.array([np.size(peakst_lp), np.size(peakst_vlp),
                     np.size(peakst_lp)/np.size(peakst_vlp)])

final_feat_zemg = np.append(np.append(features1z, features2z),features3z)
final_feat_temg = np.append(np.append(features1t, features2t),features3t)

# 12 x 2 obelezja 
