import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Dæmi um lestur fyrsta blues lagsins
(rate,sig) = wav.read("waves/blues.00000.wav")
plt.plot(np.arange(sig.shape[0])/rate,sig)
mfcc_feat = mfcc(sig,rate,nfft=1024)
# Veit ekki hvað d_mfcc og fbank eru... kom úr example.py
#d_mfcc_feat = delta(mfcc_feat, 2)
#fbank_feat = logfbank(sig,rate,nfft=1024)

#print(fbank_feat.shape)
#print(fbank_feat[1:3,:])


# Lesum inn gögnin og setjum mfcc features í fylki

(rate,sig) = wav.read("waves/blues.00000.wav")
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
        
