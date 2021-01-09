# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:06:31 2021

@author: HP
"""

import numpy as np
import pickle5 as pickle 
import pandas as pd
import tqdm
import os
import pyphysio.filters.Filters as filters
from pyphysio import EvenlySignal
from scipy.signal import find_peaks
import pyphysio.indicators.FrequencyDomain as fq_ind
import pyphysio.estimators.Estimators as estimators
#import matplotlib.pyplot as plt


def remove_drift(data,win_length):
    """
    finds low frequency drift by smoothing the data (moving average), and
    then substracts it from the original data
    """
    mask = np.ones((1,win_length))/win_length
    mask = mask[0,:]

    convolved_data = np.convolve(data,mask,'same')
    data_no_drift = data - convolved_data
    
    return data_no_drift


def gsr_feats(gsr, fs):
    """
    extracts 14 GSR features *insert citation*
    """
    gsr_no_drift = remove_drift(gsr, 256)
    gsr_sig = EvenlySignal(values = gsr_no_drift,
                           sampling_freq = fs, 
                           signal_type = 'EDA')

    gsr_lp = filters.FIRFilter([0.01,0.2],[0,0.22])(gsr_sig)
    gsr_vlp = filters.FIRFilter([0.01,0.08],[0,0.1])(gsr_sig)
    
    # extract statistical features from gsr and low-passed gsr signal
    features1 = np.array([np.mean(gsr_sig), np.std(gsr_sig), np.mean(np.diff(gsr_sig)), 
                      np.mean(np.diff(np.diff(gsr_sig))),np.mean(gsr_lp), np.std(gsr_lp), 
                      np.mean(np.diff(gsr_lp)), np.mean(np.diff(np.diff(gsr_lp)))])
    # extract zero-crossing rate from LP and very low-passed gsr signals
    features2 = np.array([np.size(np.nonzero(np.diff(gsr_lp > 0))[0]), 
                         np.size(np.nonzero(np.diff(gsr_vlp > 0))[0])])
    # extracy number of SCR peaks from both filtered gsr signals (drift removed)
    # ratio of these numbers
    # also SCR mean amplitude
    peaks_lp = find_peaks(gsr_lp, distance = 128)
    peaks_vlp = find_peaks(gsr_vlp, distance = 128)
    features3 = np.array([np.size(peaks_lp[0]), np.size(peaks_vlp[0]), 
                          np.size(peaks_lp[0])/np.size(peaks_vlp[0]), 
                          np.mean(np.append(gsr_lp[peaks_lp[0]], gsr_vlp[peaks_vlp[0]]))])
    
    final_feat = np.append(np.append(features1,features2),features3) # 14 features
    return final_feat
    
def emg_feats(emg, fs):
    """
    this function returns 12 EMG features *insert citation*
    """
    emg_sig = EvenlySignal(values = emg,
                           sampling_freq = fs, 
                           signal_type = 'EMG')
    # PSD in 4 - 40 Hz range
    feature1 = np.mean(np.log(fq_ind.InBand(4,40,'welch')(emg_sig)[1]))
    
    # extract statistical features from EMG signal
    emg_no_drift = remove_drift(emg, 256)
    features1 = np.array([feature1, np.mean(emg_no_drift), np.std(emg_no_drift), 
                          np.mean(np.diff(emg_no_drift)), 
                          np.mean(np.diff(np.diff(emg_no_drift)))])
    # extract statistical features from LP EMG signal
    emg_sig = EvenlySignal(values = emg_no_drift,
                           sampling_freq = fs, 
                           signal_type = 'EMG')
    emg_lp = filters.FIRFilter([0.01,0.3],[0,0.32])(emg_sig)
    features2 = np.array([np.mean(emg_lp), np.std(emg_lp), 
                          np.mean(np.diff(emg_lp)), np.mean(np.diff(np.diff(emg_lp)))])
    
    # extract number of myoresponses in LP and VLP EMG signals
    emg_vlp = filters.FIRFilter([0.01,0.08],[0,0.1])(emg_sig)
    peaks_lp = find_peaks(emg_lp, distance = 128)
    peaks_vlp = find_peaks(emg_vlp, distance = 128)
    features3 = np.array([np.size(peaks_lp[0]), np.size(peaks_vlp[0]),
                     np.size(peaks_lp[0])/np.size(peaks_vlp[0])])
    
    final_feat = np.append(np.append(features1,features2),features3) # 12 features
    return final_feat
    
 
def ppg_feats(ppg, fs):
    """
    this function exracts 10 HRV features from plethysmograpic data (PPG)
    *insert citation*
    """
    ppg_sig = EvenlySignal(values = ppg,
                           sampling_freq = fs, 
                           signal_type = 'PPG')

    HRV = estimators.BeatFromBP()(ppg_sig) # returns HRV in seconds
    
    # extract - first derivative mean, RMS, arc length, area perimeter ratio
    arc_length = 0
    for i in range(len(HRV)-1):
        arc_length += np.sqrt(1 + np.square(HRV[i + 1] - HRV[i]))

    arc_length /= len(HRV)

    area_per_ratio = np.sum(HRV)/len(HRV)/arc_length

    features1 = np.array([np.mean(np.diff(HRV)), np.sqrt(np.mean(np.square(HRV))),
                      arc_length, area_per_ratio])
    
    # extract mean and std of HRV
    # extract frequency domain features - PSD in low, medium and high freq bands,
    # and PSD ratio between two bands;
    HRV_sig = EvenlySignal(values = HRV,
                           sampling_freq = len(HRV)/60, 
                           signal_type = 'HRV')
    features2 = np.array([np.mean(HRV), np.std(HRV), 
                          np.mean(np.log(fq_ind.InBand(0.01,0.08,'welch')(HRV_sig)[1])), 
                      np.mean(np.log(fq_ind.InBand(0.08,0.15,'welch')(HRV_sig)[1])), 
                      np.mean(np.log(fq_ind.InBand(0.15,0.5,'welch')(HRV_sig)[1])),
                      np.mean(fq_ind.InBand(0.04,0.15,'welch')(HRV_sig)[1])
                      /np.mean(fq_ind.InBand(0.15,0.5,'welch')(HRV_sig)[1])])
    
    final_feat = np.append(features1,features2) # 10 features
    
    return final_feat
    
 
def temp_feats(temp, fs):
    """
    extracts 7 temperature features *insert citation*
    """
    # extract statistical features from the temperature signal
    features1 = np.array([np.mean(temp), np.std(temp), np.min(temp), np.max(temp),
                      np.mean(np.diff(temp))])
    
    # extract PSD in 2 frequency bands
    temp_sig = EvenlySignal(values = temp,
                           sampling_freq = fs, 
                           signal_type = 'temp')
    
    final_feat = np.append(features1, [np.mean(np.log(fq_ind.InBand(0,0.1,'welch')(temp_sig)[1])), 
                                   np.mean(np.log(fq_ind.InBand(0.1,0.2,'welch')(temp_sig)[1]))])
    # 7 features
    
    return final_feat


def standard_feat(data):
    """
    standardize features among trials (every column is one feature)
    """
    std_data = np.zeros(np.shape(data))
    for i in range(np.shape(data)[1]):
        std_data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
    return std_data


zemg_ch = 34
temg_ch = 35
GSR_ch = 36
ppg_ch = 38
temp_ch = 39
fs = 128 # Hz - all signals resampled to the same fs

for k in range(32): # number of subjects # (ovo krece od 0) - 32 subjekta
    PATH = os.path.dirname(os.path.abspath('')) + '\\data_preprocessed_python\\s{num:02d}.dat'.format(num = k + 1)

    with open(PATH, 'rb') as f:
        data = pickle.load(f, encoding = 'bytes')                                               

    # data_dim 40x40x8064 (num_videos x phy_channels x data)
    data = data[b'data']
    
    gsr_features = np.zeros((40, 14))
    zemg_features = np.zeros((40, 12))
    temg_features = np.zeros((40, 12))
    hrv_features = np.zeros((40, 10))
    temp_features = np.zeros((40, 7))

    for i in tqdm.tqdm(range(40),desc = 'Number of trials'):
    # extracting features from every trial
        signals = data[i,:,:] # signals from current trial
        gsr_features[i,:] = gsr_feats(signals[GSR_ch], fs)
        zemg_features[i,:] = emg_feats(signals[zemg_ch], fs)
        temg_features[i,:] = emg_feats(signals[temg_ch], fs)
        hrv_features[i,:] = ppg_feats(signals[ppg_ch], fs)
        temp_features[i,:] = temp_feats(signals[temp_ch], fs)
    
    # standardization of every feature
    gsr_feat = pd.DataFrame(standard_feat(gsr_features), columns = ['mean', 'std',
                            '1st derivative mean', '2nd derivative mean','LP mean', 
                            'LP std', 'LP 1st deriv mean', 'LP 2nd deriv mean', 
                            'ZCR LP', 'ZCR VLP', 'SCR LP', 'SCR VLP', 'SCR lp/vlp ratio', 
                            'SCR mean amp'])
    zemg_feat = pd.DataFrame(standard_feat(zemg_features), columns = ['sig_power4-40Hz','mean','std',
                            '1st derivative mean', '2nd derivative mean', 'LP mean', 
                            'LP std', 'LP 1st deriv mean', 'LP 2nd deriv mean', 
                            'SCR LP', 'SCR VLP', 'SCR lp/vlp ratio'])
    temg_feat = pd.DataFrame(standard_feat(temg_features), columns = ['sig_power4-40Hz','mean','std',
                            '1st derivative mean', '2nd derivative mean', 'LP mean', 
                            'LP std', 'LP 1st deriv mean', 'LP 2nd deriv mean', 
                            'SCR LP', 'SCR VLP', 'SCR lp/vlp ratio'])
    hrv_feat = pd.DataFrame(standard_feat(hrv_features), columns = ['1st deriv mean', 'RMS', 
                            'arc length', 'APR', 'mean', 'std', 'PSDlow', 'PSDmed', 'PSDhigh', 
                            'PSDratio'])
    temp_feat = pd.DataFrame(standard_feat(temp_features), columns = ['mean','std','min','max',
                             '1st deriv mean','PSD_vlp', 'PSD_lp'])
    
    gsr_feat.to_csv("gsr_features\\s{num:02d}_gsr.csv".format(num = k + 1))
    zemg_feat.to_csv("zemg_features\\s{num:02d}_zemg.csv".format(num = k + 1))
    temg_feat.to_csv("temg_features\\s{num:02d}_temg.csv".format(num = k + 1))
    hrv_feat.to_csv("hrv_features\\s{num:02d}_hrv.csv".format(num = k + 1))
    temp_feat.to_csv("temp_features\\s{num:02d}_temp.csv".format(num = k + 1))