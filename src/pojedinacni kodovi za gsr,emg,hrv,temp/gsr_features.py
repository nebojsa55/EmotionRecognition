# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 10:59:52 2020

@author: HP

"""

import numpy as np
import pickle5 as pickle 
import pandas as pd
import tqdm 
import matplotlib.pyplot as plt

GSR_ch = 36

PATH = 'C:\\Users\\HP\\Desktop\\desktop\\master predmeti\\NI projekat\\data_preprocessed_python\\s01.dat'

with open(PATH, 'rb') as f:
    data = pickle.load(f, encoding = 'bytes') 
    
    
labels = pd.DataFrame(data[b'labels'], columns = ['valence','arousal', 'dominance', 'liking'])                                               

# data je dimenzija 40x40x8064 (num_videos x phy_channels x data)
data = data[b'data']
    
#%% OVO PROMENI

for i in tqdm.tqdm(range(2),desc = 'Number of trials'):
    
    signals = data[0,:,:] # signals from current trial
    gsr_raw = signals[GSR_ch]

#%%
fs = 128
samples = 8064
time_axis = np.arange(0,samples/fs,1/fs);

plt.figure()
plt.plot(time_axis,gsr_raw)
plt.xlim([0, 65])
#plt.ylim([-9000, 5000])
plt.grid('on')    

#%% smooth-ovanje radi uklanjanja DC komponente
w = 256
mask = np.ones((1,w))/w
mask = mask[0,:]

convolved_data = np.convolve(gsr_raw,mask,'same')
gsr = gsr_raw - convolved_data
#plt.figure()
#plt.plot(gsr)

#%% filriranje GSR u dva opsega - LP i VLP
import pyphysio.filters.Filters as filters
#import ledapy
from pyphysio import EvenlySignal

gsr_sig = EvenlySignal(values = gsr,
                           sampling_freq = fs, 
                           signal_type = 'EDA')

gsr_lp = filters.FIRFilter([0.01,0.2],[0,0.22])(gsr_sig)
gsr_vlp = filters.FIRFilter([0.01,0.08],[0,0.1])(gsr_sig)


#%% feature-i gsr signala (vec uklonjena bazna linija)
# izvlace se mean, std, mean prvog i drugog izvoda = 4

features1 = np.array([np.mean(gsr_sig), np.std(gsr_sig), np.mean(np.diff(gsr_sig)), 
                      np.mean(np.diff(np.diff(gsr_sig)))])

#%% feature-i iz LP gsr signala (0.2 Hz)
# izvlace se mean, std, mean prvog i drugog izvoda = 4

features2 = np.array([np.mean(gsr_lp), np.std(gsr_lp), np.mean(np.diff(gsr_lp)), 
                      np.mean(np.diff(np.diff(gsr_lp)))])

features = np.append(features1,features2)

#%% Estimacija Phasic_component iz filtriranih GSR signala - URADJENO UKLANJANJEM DC
#import pyphysio.estimators.Estimators as estimators

#phasic_driver_lp = estimators.DriverEstim()(gsr_lp)
#phasic_driver_vlp = estimators.DriverEstim()(gsr_vlp)
phasic_driver_lp = gsr_lp
phasic_driver_vlp = gsr_vlp

#phasic_lp, _, _ = estimators.PhasicEstim(0.1)(phasic_driver_lp)
#phasic_vlp, _, _ = estimators.PhasicEstim(0.1)(phasic_driver_vlp)
#plt.plot(time_axis[0:-1],phasic_driver_lp)

#%% trazenje SCR pikova i zero-crossing-rate

zero_crosses_lp = np.nonzero(np.diff(phasic_driver_lp > 0))[0]
zero_crosses_vlp = np.nonzero(np.diff(phasic_driver_vlp > 0))[0]


# u EDA_ANALYSIS pise da se SCR pikovi traze u phasic_driver signalu 
from scipy.signal import find_peaks
peaks_lp, _ = find_peaks(phasic_driver_lp, distance = 128) #frekv opseg GSR je
                                                        # do 1Hz = 128 samples
peaks_vlp, _ = find_peaks(phasic_driver_vlp, distance = 128)

#plt.figure()
#plt.plot(phasic_driver_lp)
#plt.plot(peaks_lp, phasic_driver_lp[peaks_lp], 'rx', markersize = 7, mew = 2)
    
#srednja vrednost svih detektovanih SCR (lp i vlp)
mean_peak_val = np.mean(np.append(phasic_driver_lp[peaks_lp], 
                                  phasic_driver_vlp[peaks_vlp]))

#sledeci feature-i su: broj prolazaka kroz nulu standardizovanih signala lp i vlp,
#broj pikova u signalima lp i vlp, i odnos broja pikova + srednja vrednost gore
features1 = np.array([np.size(zero_crosses_lp),np.size(zero_crosses_vlp),
                     np.size(peaks_lp), np.size(peaks_vlp),
                     np.size(peaks_lp)/np.size(peaks_vlp), np.mean(np.append(phasic_driver_lp[peaks_lp], 
                                  phasic_driver_vlp[peaks_vlp]))])

final_feat = np.append(features,features1)  # 14 obelezja