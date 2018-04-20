
# =============================================================================
# Nota� til a� gera nokkrar myndir
# =============================================================================
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta


# Les inn besta lagi�
(rate,sig) = wav.read("CountryMan.wav")
#(rate,sig) = wav.read("waves/country.00000.wav")



sig1 = sig[10000:(10000+int(rate/2))]
mfcc1 = mfcc(sig1,rate,nfft=1024)
d1 = delta(mfcc1,2)
d2 = delta(d1,2)

plt.plot(np.arange(len(sig1))/rate, sig1)
plt.title("H�lf sek�nda �r Country man me� Axel O. & Co.")
plt.savefig("AxelWav.png")
#plt.show()

plt.imshow(mfcc1.T)
plt.title("MFCC-stu�lar hlj��b�tsins")
plt.savefig("AxelMFCC.png")
#plt.show()

plt.imshow(d1.T)
plt.title("Delta-stu�lar hlj��b�tsins")
plt.savefig("AxelDelta.png")
#plt.show()

plt.imshow(d2.T)
plt.title("Delta-delta stu�lar hlj��b�tsins")
plt.savefig("AxelDelta2.png")
#plt.show()


# Two subplots, the axes array is 1-d
f, axarr = plt.subplots(3, sharex=True)
axarr[0].imshow(mfcc1.T)
axarr[0].set_title('MFCC, Deltas and Delta-deltas')
axarr[1].imshow(d1.T)
axarr[2].imshow(d2.T)
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')

f.savefig("AxelFeatures.png")

