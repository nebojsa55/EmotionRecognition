# Emotion Recognition

Ovaj repozitorijum sadrži kod za seminarski rad "Prepoznavanje emocija na osnovu elektrofizioloških signala" koji je sastavni deo predmeta [Neuralno Inženjerstvo](http://automatika.etf.bg.ac.rs/sr/13m051ni) na [Elektrotehničkom fakultetu](https://www.etf.bg.ac.rs/)

## Organizacija repozitorijuma

Svi kodovi i izdvojena obeležja se nalaze u folderu *src*. 

1. Za izdvajanje obeležja **EEG** signala: *eeg_features.py*

2. Za izdvajanje obeležja **RESP** signala: *RESP_features.ipynb*

3. Za izdvajanje obeležja **GSR, EMG, HRV i Temp** signala: *gsr_emg_hrv_temp.py.py*

4. Catboost klasifikator: *classification.ipynb*

5. SVM klasifikator: *LinearSVM_RFE.ipynb*
 
6. Neuralne mreže: *classification_neural_network.ipynb* 

## Izdvojena obeležja

U ovom radu su inicijalno izdvojena 1490 obeležja, i navedena u sledećoj tabeli:

| Rad       | Fiziološki signal           | Obeležja  | Broj obeležja|
| :-------------: |:-------------:| :-----:|:-----:|
| Jenke 2014.      | right-aligned | $1600 ||
| Jenke 2014.      | centered      |   $12 ||
| Jenke 2014. | are neat      |    $1 ||
| Jenke 2014.      | right-aligned | $1600 ||
| po ugledu na Johnwakim 2008.      | centered      |   $12 ||
| Johnwakim 2008.| are neat      |    $1 ||
| Johnwakim 2008.    | right-aligned | $1600 ||
| DEAP   | centered      |   $12 ||
| DEAP | are neat      |    $1 ||
| Johnwakim 2008.      | right-aligned | $1600 ||
| DEAP     | centered      |   $12 ||
| Elderly 2019. | are neat      |    $1 ||
| DEAP | are neat      |    $1 ||


## Rezultati

### Najznačajnija obeležja

Na sledećoj slici su prikazana najznačajnija obeležja za klasifikaciju parametara emocije: *valence*, *arousal*, *dominance* i *liking*

![sl1](https://github.com/nebojsa55/EmotionRecognition/blob/master/pics/feature_info.png)

### Tačnost klasifikacije


### Autori
------------
* [![logo1](https://img.shields.io/github/followers/cokoladnomleko?label=Tamara%20Stajic&style=social)](https://github.com/cokoladnomleko)
* [![logo2](https://img.shields.io/github/followers/doxiekong?label=Jelena%20Jovanovic&style=social)](https://github.com/doxiekong)
* [![logo3](https://img.shields.io/github/followers/nebojsa55?label=nebojsa%20Jovanovic&style=social)](https://github.com/nebojsa55)
