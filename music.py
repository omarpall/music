import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

# Dæmi um lestur fyrsta blues lagsins, og plot
(rate,sig) = wav.read("waves/blues.00000.wav")
plt.plot(np.arange(sig.shape[0])/rate,sig)

# Lesum inn gögnin og setjum mfcc features í fylki
X = np.load("MFCC.npy")
