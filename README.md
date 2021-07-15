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
| Jenke 2014.      | EEG | Snaga, srednja vrednost (MEAN), standardna devijacija (STD), prvi izvod, normalizovan prvi izvod, drugi izvod, normalizovan drugi izvod. ||
| Jenke 2014.      | EEG      |   Aktivnost, mobilnost, kompleksnost||
| Jenke 2014. | EEG      |   Spektralna gustina snage u alfa, beta, gama i delta opsegu. ||
| Jenke 2014.      | EEG | Diskretna wavelet transformacija. ||
| po ugledu na Johnwakim 2008.      | Respiratorni pojas   |   Maksimalna amplituda spektra, MEAN spektra u opsegu 0.2-0.5 Hz, maksimalna amplituda spektralne gustine snage, MEAN spektralne gustine snage u opsegu 0.2-0.5 Hz||
| Johnwakim 2008.| HRV    |MEAN peak-to-peak intervala, STD peak-to-peak intervala, SDT prvog izvoda intervala, median intervala, opseg intervala, MEAN brzine disanja, maksimum brzine disanja, minimum brzine disanja, STD brzine disanja, broj intervala duzih od 50 i 20 ms i njihov udeo u ukupnom broju intervala, koren iz MEAN sume kvadrata peak-to-peak intervala, koeficijent promene intervala, koeficijent varijacije intervala. Ukupna gustina snage, snaga u VLF opsegu (0.003-0.04 Hz), LF opsegu (0.04-0.15 Hz) i HF opsegu (0.15-40 Hz), odnos LF/HF, normalizovana snaga u LF i HF opsegu||
| Johnwakim 2008.    | GSR | MEAN, STD, MEAN prvog izvoda, MEAN drugog izvoda nefiltriranog i lowpass (LP) filtriranog signala (fcutoff=0.2 Hz);
Broj pikova u LP i very lowpass (VLP) signalu (fcutoff=0.08 Hz), njihov odnos i srednja amplitude.||
| DEAP   | GSR     |  Brzina prolaska kroz nulu LP i VLP signala. ||
| DEAP | zEMG + tEMG    |   Snaga u frekvencijskom opsegu [4, 40] Hz||
| Johnwakim 2008.      | MEAN, STD, MEAN prvog izvoda, MEAN drugog izvoda nefiltriranog i LP filtriranog signala (fcutoff=0.3 Hz);
Broj pikova u LP i VLP signalu (fcutoff=0.1 Hz) i njihov odnos.
||
| DEAP     | HRV          | MEAN, STD, odnos snage u opsezima [0.04, 0.15] Hz i [0.15, 0.5] Hz.
Snaga u tri opsega – [0.01,0.08] , [0.08,0.15], [0.15, 0.5]   Hz
||
| Elderly 2019. | HRV     |  MEAN prvog izvoda, duzina luka (arc-length), RMS, obim podrucija (area-perimeter ratio)
 ||
| DEAP | Telesna temperatura     |    MEAN, STD, MEAN prvog izvoda, minimalna i maksimalna temperatura, snaga u opsezima:
[0, 0.1] Hz i [0.1, 0.2] Hz
||


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
