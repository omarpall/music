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

flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
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
model.add(Conv2D(32, (100, 6), activation='relu', input_shape=(2985, 13, 1)))
model.add(Conv2D(32, (100, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#model.add(Conv2D(64, (100, 6), activation='relu'))
#model.add(Conv2D(64, (100, 6), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
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

X_train = np.load("XTrainButar.npy")
y_train = np.load("yTrainButar.npy")
for i in range(len(y_train)):
    y_train[i] = flokkar.index(y_train[i])
y_train = keras.utils.to_categorical(y_train,num_classes=10)

model = Sequential()

