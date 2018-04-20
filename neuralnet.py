import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# =============================================================================
# Tauganet á lagabútum - Notum VGG-like convnet á keras.io
# =============================================================================

flokkar = ["blues","classical","country","disco","hiphop",
           "jazz","metal","pop","reggae","rock"]

# Loading the data
X_train = np.load("XTrainButar.npy")
y_train = np.load("yTrainButar.npy")

# Koma labels á one-hot form
# T.d. verður "classical" að [0,1,0,0,0,0,0,0,0,0]
for i in range(len(y_train)):
    y_train[i] = flokkar.index(y_train[i])
y_train = keras.utils.to_categorical(y_train,num_classes=10)

# Splitta training settinu upp í training og validation mengi
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1000, random_state=42)

# Koma gögnunum á rétt form fyrir Conv2D
X_train = X_train.reshape(7000,-1,13,1)
X_val = X_val.reshape(1000,-1,13,1)

# Smíðum módelið. Við prófuðum alls konar layers 
# og þetta virkaði best af því sem við prófuðum.
# Þetta er mjög svipað VGG-like convnet á keras.io
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(298,13,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

# Notum Stochastic Gradient Descent
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])

# Þjálfum tauganetið með batch_size=100 í 10 epochs.
model.fit(X_train, y_train, batch_size=100, epochs=10,
          validation_data=(X_val,y_val))

# =============================================================================
# Testum módelið
# =============================================================================

# Hlöðum test settinu
# Hver lína eru 10 þriggja sek bútar af MFCC (sjá saveFeatures.py)
X_test = np.load("XTestButar.npy")
y_test_original = np.load("yTestButar.npy")

# Gerum y_test fyrir lagabútana, þar sem hverju lagi er skipt í 10 búta
y_test = []
for i in range(len(y_test_original)):
    for _ in range(10):
        y_test.append(y_test_original[i])
y_test = np.asarray(y_test)

# Breyta því í one-hot
for i in range(len(y_test)):
    y_test[i] = flokkar.index(y_test[i])
y_test = keras.utils.to_categorical(y_test,num_classes=10)

# Spáum á bútunum
X_test_prepared = X_test.reshape(-1,298,13,1)
ypred_prepared = model.predict(X_test_prepared)

# Leggjum saman softmax outputtin fyrir bútana
# (ATH þetta er ekki það sama og majority vote, en samt næstum)
ypred = np.zeros((200,10))
for i in range(200):
    for j in range(10): # Tíu bútar
        ypred[i] = ypred[i]+ypred_prepared[10*i+j]
# Veljum stærstu summuna sem spágildið
finalpred = []
for i in range(len(ypred)):
    finalpred.append(flokkar[np.argmax(ypred[i])])
finalpred = np.asarray(finalpred)

# Finnum testing accuracy
acc = np.average(finalpred==y_test_original)
print("We get a testing accuracy of",acc)

# Reiknum confusion matrixið
# Rétt gildi eru línur, spáða gildi eru dálkar
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test_original,finalpred, labels=flokkar) 
#print(cm1)
plt.matshow(cm1, cmap="hot")
plt.xticks(range(len(flokkar)), flokkar, rotation=80)
plt.yticks(range(len(flokkar)), flokkar)
plt.colorbar()
plt.savefig("cmNeuralNetwork.png")
#plt.show()

cm1_hlutf = cm1/cm1.sum(axis=1,keepdims=True)
#print(cm1_hlutf)
plt.matshow(cm1_hlutf, cmap="hot")
plt.xticks(range(len(flokkar)), flokkar, rotation=80)
plt.yticks(range(len(flokkar)), flokkar)
plt.colorbar()
plt.savefig("cmHlutfallsNeuralNetwork.png")
#plt.show()

# Spánákvæmni hvers flokks:
print(np.diagonal(cm1_hlutf))

# =============================================================================
# Testing á lögum sem við völdum
# =============================================================================
import scipy.io.wavfile as wav
from python_speech_features import mfcc


def spalag(model, filepath):
    (rate,sig) = wav.read(str(filepath))
    sig = sig[0:660000] # Minnsta lagið er 660000 punktar
    countryman = []
    for i in range(10):
        tmp = sig[i*66000:(i+1)*66000]
        mfcc_feat = mfcc(tmp,rate,nfft=1024).reshape(-1)
        countryman.append(mfcc_feat)
    countryman = mfcc_feat.reshape(-1,298,13,1) 
    AxelO = model.predict(countryman)
    #print(AxelO)
    return(flokkar[np.argmax(AxelO)])

print("Höfundur       | Lag               | Flokkur    | Spá")
print(" Axel O. & Co. |  Country man      |  country   | ",spalag(model,"CountryMan.wav"))
print(" Axel O. & Co. |  You Are Trouble  |  country   | ",spalag(model,"YouAreTrouble.wav"))
print(" Skálmöld      |  Kvaðning         |  metal     | ",spalag(model,"skalmold.wav"))
print(" Amabadama     |  Hossa Hossa      |  reggae    | ",spalag(model,"amabadama.wav"))
print(" Howard Shore  |  LotR Shire theme |  classical | ",spalag(model,"shire.wav"))



