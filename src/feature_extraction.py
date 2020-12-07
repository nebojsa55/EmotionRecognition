"""
U ovoj skripti se ucitavaju pretprocesirani signali koji su dati u okviru
baze i u pickle formatu. Potrebno je instalirati pickle biblioteku ako je nemate
Zadatak ove skripte je da se izdvoje obelezja od mogucih signala kako bi se u sledecem
koraku izvrsila klasifikacija
"""

import numpy as np
import pickle 
import pandas as pd
import os.path

# Indeksi signala za promenljivu data
# Ako zelite da izdvojite samo EEG npr, notacija je kao
# data[x,EEG_ch,:] gde je x broj triala

EEG_ch = range(32);
EOG_ch = range(32,34);
EMG_ch = range(34,36);
GSR_ch = 36;
Resp_ch = 37; # respiracija
Plet_ch = 38; # pletizmograf
Temp_ch = 39; # temperatura

PATH =  os.path.dirname(__file__) + '\..\dataset\signals_processed\DEAP\s01.dat'

with open(PATH, 'rb') as f:
    data = pickle.load(f, encoding = 'bytes') 
    # Ako koristite Python 3 potrebno je staviti encoding = 'byte' zato sto je 
    # picklovanje vrseno u Python 2
    
    
labels = pd.DataFrame(data[b'labels'], columns = ['valence','arousal', 'dominance', 'liking'])                                               
data = data[b'data']


##### FEATURE EXTRACTION matrica 40 x num_of_features ########
#### Ovde dodati vas kod #######3


#### 

