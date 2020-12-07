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

