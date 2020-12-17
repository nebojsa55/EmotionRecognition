"""
Implementirane su funkcije za izracunavanje obelezja iz 2. i 4. reda
iz Tamarine tabele
Koriscena je biblioteka pyphysio, toplo preporucujem posto su mnoge funkcije
vec napisane, i to jos za ekg,emg i gsr pored eeg signala

"""

import numpy as np
from pyphysio import EvenlySignal

def statistical_features(eeg,fs):
    
    """
    Function will compute the most used statistical features of eeg signal

    Input: 
        eeg -> 1 channel eeg signal
        fs -> sampling frequency

    """
    import pyphysio.indicators.TimeDomain as td_ind
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

    Input: 
        eeg -> 1 channel eeg signal
        fs -> sampling frequency

    """
    import pyphysio.indicators.FrequencyDomain as fq_ind
    import pyphysio.filters.Filters as filters
    
    eeg_sig = EvenlySignal(values = eeg,
                           sampling_freq = fs, 
                           signal_type = 'eeg')
    
    
    # Konstrukcija filtara 
    delta_filt_ = filters.FIRFilter([0.5, 4.2],[0.1,5])
    theta_filt_ = filters.FIRFilter([3.8,8.2],[3,9])
    alpha_filt_ = filters.FIRFilter([7.8,12.2], [7,13])
    beta_filt_ = filters.FIRFilter([15,32], [14,33])
    gamma_filt_ = filters.FIRFilter([32,60], [31, 61])
    
    
    # Izdvajanje talasa
    delta = delta_filt_(eeg_sig)
    theta = theta_filt_(eeg_sig)
    alpha = alpha_filt_(eeg_sig)
    beta = beta_filt_(eeg_sig)
    gamma = gamma_filt_(eeg_sig)
    
    # Statisticka obelezja
    delta_statistical = statistical_features(delta, fs)
    theta_statistical = statistical_features(theta, fs)
    alpha_statistical = statistical_features(alpha, fs)
    beta_statistical = statistical_features(beta, fs)
    gamma_statistical = statistical_features(gamma, fs) 
    
    # PSD
    
    delta_PSD = fq_ind.InBand(0.5,4,'welch')(eeg_sig)
    theta_PSD = fq_ind.InBand(4,8,'welch')(eeg_sig)
    alpha_PSD = fq_ind.InBand(8,12,'welch')(eeg_sig)
    beta_PSD = fq_ind.InBand(15,32,'welch')(eeg_sig)
    gamma_PSD = fq_ind.InBand(32,60,'welch')(eeg_sig)
    
    
    ## Create dictionary
    
    final_dict = {}
    
    for key in delta_statistical:
        final_dict.update({'delta_' + key : delta_statistical[key]})
    for key in theta_statistical:
        final_dict.update({'theta_' + key : theta_statistical[key]})
    for key in alpha_statistical:
        final_dict.update({'alpha_' + key : alpha_statistical[key]})
    for key in beta_statistical:
        final_dict.update({'beta_' + key : beta_statistical[key]})
    for key in gamma_statistical:
        final_dict.update({'gamma_' + key : gamma_statistical[key]})
    
    final_dict.update({'delta_PSD' : np.mean(np.log(delta_PSD[1]))})
    final_dict.update({'theta_PSD' : np.mean(np.log(theta_PSD[1]))})
    final_dict.update({'alpha_PSD' : np.mean(np.log(alpha_PSD[1]))})
    final_dict.update({'beta_PSD' :  np.mean(np.log(beta_PSD[1]))})
    final_dict.update({'gamma_PSD' : np.mean(np.log(gamma_PSD[1]))})
    
    return final_dict
        
        
if __name__ == "__main__":
    
    # """
    # Currently the pyphysio library does not contain test EEG data
    # so we will generate some artificial data for testing.
    # Feel free to insert here your own eeg data and to test
    # the functions
    # """
    
    # np.random.seed(4)    
    # data = np.random.uniform(0,350,8000)
    # fs = 200
    
    # Treba ovaj fajl prvo kreirati skriptom izdvajanje_test_signala
    data = np.loadtxt('eeg_test.txt')
    fs = 128 #Hz
    
    
    # Dictionaries
    stats = statistical_features(data,fs)
    band = band_features(data, fs)
        
    
    
    
    
