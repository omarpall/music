# music
Þið veljið eitt af verkefnunum hér að neðan og leysið í 2ja manna hópum. Aðferðafræði og helstu niðurstöður takið þið saman í stuttri skýrslu (hámark 10 bls. að viðauka undanskildum) sem á að skila í síðasta lagi föstudaginn 20.4. Verkefnalýsingum er ætlað að koma ykkur af stað en ykkur er frjálst að bæta við þær. Notið numpy og scikit-learn pakkana eins og við á. Nánari upplýsingar um skýrsluna er að finna aftar í skjalinu. Þau ykkar sem velja verkefni #1 og #2 skuldbinda sig til að skoða ekki lausnir frá öðrum (sem finna má á netinu).

## 1. Flokkun á tónlist
Markhópur: Allir.
Aðferðafræði: Forvinnsla gagna, smíði auðkenna, flokkun, val á líkani.

Þetta verkefni snýst um læra að þekkja mismunandi tegundir tónlistar út frá hljóðupptökum. GTZAN gagnasafnið inniheldur 1000 hljóðbúta, 30 sek hver, sem safnað var á 22050 Hz (16-bitar, mono). Í safninu eru 10 mismunandi tegundir tónlistar og eru 100 bútar í hverjum. Tegundirnar sem um ræðir eru diskó, þungarokk, popp, hipp hopp, rokk, blús, klassísk tónlist, kantrí, jazz og reggí.

Þið þurfið að reikna auðkenni (e. features) fyrir hljóðupptökurnar. Hér á að nota svokallaða Mel frequency cepstral stuðla (MFCC) og stærðir sem eru leiddar af þeim ("delta" og "delta-deltas"), sjá MFCC kynningu sem vísað er í hér að neðan. Byrjið með tvo flokka, fáar kennistærðir og einfalda flokkara og vinnið ykkur svo “upp”. Mögulega er erfitt að greina suma flokkana í sundur. Ef svo er getið þið reynt að fækka flokkunum eitthvað. Þið getið líka athugað með að nota aðrar kennistærðir. Skoðið gögnin myndrænt til að átta ykkur betur á verkefninu. Metið nákvæmni flokkara með hefðbundnum aðferðum en til viðbótar við flokkunarnákvæmni getið þið skoðað "precision", "recall" og "F-score". Í lokin megið þið gjarnan prófa besta flokkarann ykkar á nokkrum lögum sem þið veljið sjálf.


### Athugasemdir:

1. GTZAN gagnasafnið: http://marsyasweb.appspot.com/download/data_sets/
Skrárnar eru á .au sniði og þarf að byrja á að breyta þeim yfir á .wav snið. Það má t.d. gera með Sound eXchange pakkanum: http://sox.sourceforge.net/
Þennan pakka má nota til að koma lögum í prófunarsettinu ykkar á sama form og þjálfunargögnin.
2. Mel Frequency Cepstral Coefficient (MFCC)
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency- cepstral-coefficients-mfccs/
3. Python pakki sem reiknar MFCC: https://github.com/jameslyons/python_speech_features
Sjá examply.py. MFCC stuðlarnir eru tímaháðir og er skilað í formi fylkis sem hefur u.þ.b. 3000 raðir og 13 dálka (fleiri dálkar bætast við þegar þið reiknið “delta” og “delta-deltas”). Fylkin má fletja út í vigra svipað og gert er með myndir.
