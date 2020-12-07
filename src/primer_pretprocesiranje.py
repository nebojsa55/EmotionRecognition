"""
Pomocu ove skripte se ucitavaju svi raw signali i vrsi nad njima pretprocesiranje.
Probacemo kao sto smo se dogovorili, da uradimo prvo klasifikaciju nad podacima 
koji su pretprocesirani, pa kasnije mozemo npr da uporedimo rezultate sa onima
kad mi odradimo taj korak. 

Ucitavanje raw signala se vrsi pomocu biblioteke pyedflib tako da je prvo potrebno
prvo instalirati tu biblioteku u anaconda shellu pomocu komande 
pip install pyEDFlib
"""

import numpy as np
import pandas as pd
import os.path
import pyedflib as edf

# Brojevi kanala za sve signale
EEG_ch = range(32);
EOG_ch = range(32,36);
EMG_ch = range(36,40);
GSR_ch = 40;
Resp_ch = 44; # respiracija
Plet_ch = 45; # pletizmograf
Temp_ch = 46; # temperatura


# Ucitavanje raw signala (primer za prvog subjekta)
PATH_SIGNALS = os.path.dirname(__file__) + '\..\dataset\signals_raw\s01.bdf'

data = edf.EdfReader(PATH_SIGNALS)

# Ukupan broj signala
num_signals = data.signals_in_file

# Nazivi signala
signal_labels = data.getSignalLabels()

# Perioda odabiranja
fs = data.getSampleFrequencies()[0] # ista je za sve, trebace neke signale downsamplovati
N = data.getNSamples()[0];

######## Izdvajanje svih signala ##########3
EEG = np.zeros((len(EEG_ch),N))
for i in EEG_ch:
    EEG[i,:] = data.readSignal(i)
    
EOG = np.zeros((len(EOG_ch),N))
for i in EOG_ch:
    EOG[i-32,:] = data.readSignal(i)
    
EMG = np.zeros((len(EMG_ch),N))
for i in EMG_ch:
    EMG[i-36,:] = data.readSignal(i)
    
GSR = data.readSignal(GSR_ch)
Resp = data.readSignal(Resp_ch)
Plet = data.readSignal(Plet_ch)
Temp = data.readSignal(Temp_ch)



##### Prikaz nekog signala ###########
import matplotlib.pyplot as plt

t = np.linspace(0,N/fs,N)

plt.figure()
plt.plot(t,EEG[0,:])
plt.xlabel('Vreme [s]')


####### Ispod vrsiti pretprocesiranje i posle sacuvati u pickle formatu npr #######

   


data.close()

