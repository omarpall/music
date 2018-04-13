# -*- coding: utf-8 -*-
import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
#from python_speech_features import logfbank
import scipy.io.wavfile as wav

MFCC = []
d_MFCC = []
dd_MFCC = []
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
for flokkur in flokkar:
    print(flokkur)
    for i in range(100):
        file = "waves/"+flokkur+"."+str(i).zfill(5)+".wav"
        (rate,sig) = wav.read(file)
        sig = sig[0:660000] # Minnst lagið er 660000 punktar
        mfcc_feat = mfcc(sig,rate,nfft=1024)
        d_mfcc_feat = delta(mfcc_feat,2)
        dd_mfcc_feat = delta(d_mfcc_feat,2)
        MFCC = np.append(MFCC, mfcc_feat.flatten())
        d_MFCC = np.append(d_MFCC, d_mfcc_feat.flatten())
        dd_MFCC = np.append(dd_MFCC, dd_mfcc_feat.flatten())
MFCC = MFCC.reshape((1000,38805))        
d_MFCC = d_MFCC.reshape((1000,38805))        
dd_MFCC = dd_MFCC.reshape((1000,38805))        

# Vista mfcc features sem MFCC.npy
np.save("MFCC.npy", MFCC)
np.save("d_MFCC.npy", d_MFCC)
np.save("dd_MFCC.npy", dd_MFCC)
