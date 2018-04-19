import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D
import matplotlib.pyplot as plt

# =============================================================================
# Neural network on full songs - VGG-like convnet á keras.io
# =============================================================================
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
# Load the MFCC features
X = np.load("MFCC.npy")

flokkar = ["blues","classical","country","disco","hiphop",
           "jazz","metal","pop","reggae","rock"]
y = []
for i in range(10):
    tmp = np.zeros(10)
    tmp[i]=1
    for _ in range(100):
        y.append(tmp)
y = np.asarray(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=42)

X_train = X_train.reshape(800,-1,13,1)
X_test = X_test.reshape(200,-1,13,1)

model = Sequential()
# ATH þetta sökkar. <0.1 eftir 3 epochs. BREYTA!!
# input: 2985x13 images with 1 channel -> (2985,13,1) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (10, 2), activation='relu', input_shape=(2985, 13, 1)))
model.add(Conv2D(32, (10, 2), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

#model.add(Conv2D(64, (100, 6), activation='relu'))
#model.add(Conv2D(64, (100, 6), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dense(256, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=50, epochs=10)


score = model.evaluate(X_test, y_test, batch_size=50)

print(score)


# =============================================================================
# Neural network on reduced songs
# =============================================================================

# Loading the data
X_train = np.load("XTrainButar.npy")
y_train = np.load("yTrainButar.npy")

for i in range(len(y_train)):
    y_train[i] = flokkar.index(y_train[i])
y_train = keras.utils.to_categorical(y_train,num_classes=10)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=1000, random_state=42)

X_train = X_train.reshape(7000,-1,13,1)
X_val = X_val.reshape(1000,-1,13,1)


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(298,13,1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 1)))
#model.add(Dropout(0.75))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', 
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=100, epochs=10,
          validation_data=(X_val,y_val))



# Testing on the test set.
X_test = np.load("XTestButar.npy")
y_test_original = np.load("yTestButar.npy")

# Hvert lag eru 10 þriggja sek bútar af MFCC
y_test = []
for i in range(len(y_test_original)):
    for _ in range(10):
        y_test.append(y_test_original[i])
y_test = np.asarray(y_test)

# Breyta í one-hot
for i in range(len(y_test)):
    y_test[i] = flokkar.index(y_test[i])
y_test = keras.utils.to_categorical(y_test,num_classes=10)


# Predicta bitana
X_test_prepared = X_test.reshape(-1,298,13,1)
ypred_prepared = model.predict(X_test_prepared)

# Leggjum saman þetta softmax dót og veljum stærsta sem predictið
# (ATH þetta er ekki það sama og majority vote, en samt næstum)
ypred = np.zeros((200,10))
for i in range(200):
    for j in range(10): # Tíu bútar
        ypred[i] = ypred[i]+ypred_prepared[10*i+j]

finalpred = []
for i in range(len(ypred)):
    finalpred.append(flokkar[np.argmax(ypred[i])])
finalpred = np.asarray(finalpred)

# Majority vote
majorityPred = []
for i in range(200):
    tmp = np.zeros(10)
    for j in range(10): # Tíu bútar
        maxarg = np.argmax(ypred_prepared[10*i+j])
        tmp[maxarg] = tmp[maxarg] + 1
    prediction = np.random.choice(np.flatnonzero(tmp == np.max(tmp)))
    majorityPred.append(flokkar[prediction])
majorityPred = np.asarray(majorityPred)


acc = np.average(finalpred==y_test_original)
print("We get a testing accuracy of",acc)
accmaj = np.average(majorityPred==y_test_original)
print("We get a testing majority accuracy of",acc)


from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test_original,finalpred, labels=flokkar) 
print(cm1)


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



