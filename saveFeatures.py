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

# Vista mfcc features sem MFCC.npy, og deltas og deltadeltas
np.save("MFCC.npy", MFCC)
np.save("d_MFCC.npy", d_MFCC)
np.save("dd_MFCC.npy", dd_MFCC)

# Labels fyrir lögin
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]
    
# Búta hvert lag í 10 búta
trainMFCC = []
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
for flokkur in flokkar:
    print(flokkur)
    for i in range(100):
        file = "waves/"+flokkur+"."+str(i).zfill(5)+".wav"
        if(i%25==0):
            print(file)
        (rate,sig) = wav.read(file)
        sig = sig[0:660000] # Minnst lagið er 660000 punktar
        # 10 bútar
        for i in range(10):
            mfcc_butur = mfcc(sig[i*66000:(i+1)*66000],rate,nfft=1024)
            trainMFCC = np.append(trainMFCC,mfcc_butur.flatten())
trainMFCC = trainMFCC.reshape(1000,-1)

# Split songs into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainMFCC, y, test_size=200, random_state=42)

X_train = X_train.reshape(-1,3874)
yy_train = []
for flokk in y_train:
    for _ in range(10):
        yy_train = np.append(yy_train,flokk)
        

np.save("XTrainButar.npy",X_train)
np.save("yTrainButar.npy",yy_train)
np.save("XTestButar.npy",X_test)
np.save("yTestButar.npy",y_test)


