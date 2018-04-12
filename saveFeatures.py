# -*- coding: utf-8 -*-
import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav

MFCC = []
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
for flokkur in flokkar:
    print(flokkur)
    for i in range(100):
        file = "waves/"+flokkur+"."+str(i).zfill(5)+".wav"
        (rate,sig) = wav.read(file)
        sig = sig[0:660000] # Minnst lagið er 660000 punktar
        mfcc_feat = mfcc(sig,rate,nfft=1024)
        MFCC = np.append(MFCC, mfcc_feat.flatten())
MFCC = MFCC.reshape((1000,38805))        

# Vista mfcc features sem MFCC.npy
np.save("MFCC.npy", MFCC)
