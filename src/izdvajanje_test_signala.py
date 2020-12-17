"""

Ova jednostavna skripta sluzi samo da se izdvoje test signali od prvog ispitanika
kako bi lakse debuggovali kod sa njima posle, i nije bitna za projekat.
TODO : izdvojiti gsr,temperaturu,respiraciju
"""

import numpy as np
import os.path
import pickle

PATH =  os.path.dirname(__file__) + '\..\dataset\signals_processed\DEAP\s01.dat'

with open(PATH, 'rb') as f:
    data = pickle.load(f, encoding = 'bytes') 
    # Ako koristite Python 3 potrebno je staviti encoding = 'byte' zato sto je 
    # picklovanje vrseno u Python 2
    
eeg = data[b'data'][0,0,:]

np.savetxt('eeg_test.txt', eeg)
    