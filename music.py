import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav


# Dæmi um lestur fyrsta blues lagsins
(rate,sig) = wav.read("waves/blues.00000.wav")
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat.shape)
print(fbank_feat[1:3,:])
