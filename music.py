import numpy as np
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Dæmi um lestur fyrsta blues lagsins, og plot
(rate,sig) = wav.read("waves/blues.00000.wav")
plt.plot(np.arange(sig.shape[0])/rate,sig)

# Lesum inn gögnin og setjum mfcc features í fylki
X = np.load("MFCC.npy")
# Setjum rétt labels í y vigur
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


