"""
Implementirane su funkcije za izracunavanje obelezja iz 2., 3. i 4. reda
iz Tamarine tabele
Koriscena je biblioteka pyphysio, toplo preporucujem posto su mnoge funkcije
vec napisane, i to jos za ekg,emg i gsr pored eeg signala

"""

import numpy as np
from pyphysio import EvenlySignal

def statistical_features(eeg,fs):
    """
    Function will compute the most used statistical features of eeg signal

    Parameters
    ----------
    eeg : array,list -> 1-channel eeg signal
    fs : int -> sample frequency

    Returns
    -------
    stats : dic -> dictionary containing the most used statistical EEG features

    """
    eeg_sig = EvenlySignal(values = eeg,
                           sampling_freq = fs, 
                           signal_type = 'eeg')
    
    # Normalization of the eeg signal using minmax method
    eeg_sig_norm = (eeg_sig - np.min(eeg_sig)) / (np.max(eeg_sig)-np.min(eeg_sig))
    
    mean = np.mean(eeg_sig) # raw signal mean
    std = np.std(eeg_sig) # raw signal std
    mean_norm = np.mean(eeg_sig_norm) # norm signal mean
    std_norm = np.std(eeg_sig_norm) # norm signal std 
    
    from pyphysio import Diff
    
    # Mean of the absolute values of the first differences - raw signal
    diff_ = Diff(degree = 1)
    diff2_ = Diff(degree = 2)
    mean_abs_first_diff = np.mean(np.abs(diff_(eeg_sig)))
    mean_abs_second_diff = np.mean(np.abs(diff2_(eeg_sig)))
    
    # Mean of the absolute values of the first differences - norm signal
    mean_abs_first_diff_norm = np.mean(np.abs(diff_(eeg_sig_norm)))
    mean_abs_second_diff_norm = np.mean(np.abs(diff2_(eeg_sig_norm)))
    
    
    # Return a dictionary
    return {
            'mean_raw' : mean,
            'std_raw' : std,
            'mean_norm' : mean_norm,
            'std_norm' : std_norm,
            'mean_first_diff_raw' : mean_abs_first_diff,
            'mean_second_diff_raw' : mean_abs_second_diff,
            'mean_first_diff_norm' : mean_abs_first_diff_norm,
            'mean_second_diff_norm' : mean_abs_second_diff_norm        
           }
    
def band_features(eeg,fs):
    """
    Function will compute the most used statistical features of eeg signal
    for every bandwave as well as PSD.

    Parameters
    ----------
    eeg : array,list -> 1-channel eeg signal
    fs : int -> sample frequency

    Returns
    -------
    final_dict : dic -> dictionary containing bandwave features

    """
    import pyphysio.indicators.FrequencyDomain as fq_ind
    import pyphysio.filters.Filters as filters
    
    eeg_sig = EvenlySignal(values = eeg,
                           sampling_freq = fs, 
                           signal_type = 'eeg')
    
    
    # Konstrukcija filtara 
    
    theta_filt_ = filters.FIRFilter([3.8,8.2],[3,9])
    alpha_filt_ = filters.FIRFilter([7.8,12.2], [7,13])
    beta_filt_ = filters.FIRFilter([15,32], [14,33])
    
    
    # Izdvajanje talasa
    
    theta = theta_filt_(eeg_sig)
    alpha = alpha_filt_(eeg_sig)
    beta = beta_filt_(eeg_sig)
    
    # Statisticka obelezja

    theta_statistical = statistical_features(theta, fs)
    alpha_statistical = statistical_features(alpha, fs)
    beta_statistical = statistical_features(beta, fs) 
    
    # PSD
    
    theta_PSD = fq_ind.InBand(4,8,'welch')(eeg_sig)
    alpha_PSD = fq_ind.InBand(8,12,'welch')(eeg_sig)
    beta_PSD = fq_ind.InBand(15,32,'welch')(eeg_sig)
    
    
    ## Create dictionary
    
    final_dict = {}
    
    for key in theta_statistical:
        final_dict.update({'theta_' + key : theta_statistical[key]})
    for key in alpha_statistical:
        final_dict.update({'alpha_' + key : alpha_statistical[key]})
    for key in beta_statistical:
        final_dict.update({'beta_' + key : beta_statistical[key]})
    

    final_dict.update({'theta_PSD' : np.mean(np.log(theta_PSD[1]))})
    final_dict.update({'alpha_PSD' : np.mean(np.log(alpha_PSD[1]))})
    final_dict.update({'beta_PSD' :  np.mean(np.log(beta_PSD[1]))})

    
    return final_dict

def hjorth_features(eeg,fs):
    """
    Calculation of EEG features introduced by Hjorth in "B. Hjorth, 
    “EEG analysis based on time domain properties,” Electroencephalogr.
    Clin. Neurophysiol., vol. 29, no. 3, pp. 306–310, 1970."
    
    Parameters
    ----------
    eeg : array,list -> 1-channel eeg signal
    fs : int -> sample frequency

    Returns
    -------
    hjorth : dic -> dictionary containing activity,mobility and complexity parameters

    """
    eeg_sig = EvenlySignal(values = eeg,
                           sampling_freq = fs, 
                           signal_type = 'eeg')      
    
    # activity is just squared std!
    a = pow(np.sum(eeg_sig - np.mean(eeg_sig)),2)/len(eeg_sig)
    m = np.sqrt(np.var(np.gradient(eeg_sig))/np.var(eeg_sig))
    c = np.sqrt(np.var(np.gradient(np.gradient(eeg_sig)))/np.var(np.gradient(eeg_sig)))/np.sqrt(np.var(np.gradient(eeg_sig))/np.var(eeg_sig))
    
    return {
        "activity" : a,
        "mobility" : m,
        "complexity": c
           }
    

def SampEn(L, m, r):
    """
    Helper function. Implementation for approximate entropy given on 
    hhttps://en.wikipedia.org/wiki/Sample_entropy and is based on the paper 
    "Approximate Entropy and Sample Entropy: A Comprehensive Tutorial" (2019) Delgado-Bonal et al.

    """
    N = len(L)
    B = 0.0
    A = 0.0
    
    
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B) 
  
def discrete_wavelet_features(eeg,fs):
    """
    Calculation of EEG features by using DWT (discrete wavelet transform). The
    features computed in this function are described in the paper "Feature Extraction 
    and Selection for Emotion Recognition from EEG" (2014) Jenke et al.
    
    Parameters
    ----------
    eeg : array,list -> 1-channel eeg signal
    fs : int -> sample frequency

    Returns
    -------
    dwt : dic -> dictionary containing various DWT EEG features

    """
    
    import pywt
    
    num_of_levels = np.floor(np.log2(fs/8))
    coeffs = pywt.wavedec(eeg, 'db4', level = int(num_of_levels))
    
    alpha = coeffs[2]
    beta = coeffs[3]
    gamma = coeffs[4]
    
    # Calculate energy
    alpha_energy = np.sum(alpha**2)
    beta_energy = np.sum(beta**2)
    gamma_energy = np.sum(gamma**2)
    
    total = alpha_energy + beta_energy + gamma_energy
    
    # Calculate recursive energy efficienc(see Jenke 2014)
    ree_alpha = alpha_energy/total
    ree_beta = beta_energy/total
    ree_gamma = gamma_energy/total
    
    # Calculate entropy
    alpha_entropy = SampEn(alpha,2,3)
    beta_entropy = SampEn(beta,2,3)
    gamma_entropy = SampEn(gamma,2,3)

    return {
        "alpha_energy" : alpha_energy,
        "beta_energy" : beta_energy,
        "gamma_energy" : gamma_energy,
        "alpha_ree" : ree_alpha,
        "beta_ree" : ree_beta,
        "gamma_ree" : ree_gamma,
        "alpha_entropy" : alpha_entropy,
        "beta_entropy" : beta_entropy,
        "gamma_entropy" : gamma_entropy
           }
    

        
if __name__ == "__main__":
    
    import pandas as pd 
    import os,tqdm
    import pickle 
    
    # Extract features from EEG signals
    EEG_ch = range(32)
    EEG_ch_names = ['Fp1','AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3','T7','CP5','CP1',
                    'P3','P7','PO3','O1','Oz','Pz','Fp2','AF4','Fz','F4','F8','FC6',
                    'FC2','Cz','C4','T8','CP6','CP2','P4','P8','PO4','O2']
    fs = 128 # Hz
    
    for j in tqdm.tqdm(EEG_ch,desc = 'EEG channel number', total = 32):
        d = []
        for k in tqdm.tqdm(range(32),desc = 'Subject ID', total = 32): 
            PATH = os.path.dirname(os.path.abspath('')) 
            + '\dataset\signals_processed\DEAP\s{num:02d}.dat'.format(num = k + 1)
            
            with open(PATH, 'rb') as f:
                data = pickle.load(f, encoding = 'bytes') # data_dim 40x40x8064 (num_videos x phy_channels x data)                                             
        

            data = data[b'data']
            d1 = pd.DataFrame([statistical_features(data[i,j,:],fs) 
                            for i in tqdm.tqdm(range(40), desc = 'Stat features', total = 40)])
            d1 = (d1-d1.mean())/d1.std() #standardization
            d2 = pd.DataFrame([band_features(data[i,j,:],fs) 
                            for i in tqdm.tqdm(range(40), desc = 'Band features', total = 40)])
            d2 = (d2-d2.mean())/d2.std() #standardization
            d3 = pd.DataFrame([hjorth_features(data[i,j,:],fs) 
                            for i in tqdm.tqdm(range(40), desc = 'Hjorth features', total = 40)])
            d3 = (d3-d3.mean())/d3.std() #standardization
            d4 = pd.DataFrame([discrete_wavelet_features(data[i,j,:], fs) 
                            for i in tqdm.tqdm(range(40), desc = 'Wavelet features', total = 40)])
            d4 = (d4-d4.mean())/d4.std() #standardization
            # Put together all the features and save them
            d.append(pd.concat([d1,d2,d3,d4], axis = 1))
        
     
            