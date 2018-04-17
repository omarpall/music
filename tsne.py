# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:58:02 2018

@author: notandi
"""

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
# Dæmi um lestur fyrsta blues lagsins, og plot
#(rate,sig) = wav.read("waves/blues.00000.wav")
#plt.plot(np.arange(sig.shape[0])/rate,sig)

# Read the data (mfcc features, deltas and delta-deltas)
MFCC = np.load("MFCC.npy")
d_MFCC = np.load("d_MFCC.npy")
dd_MFCC = np.load("dd_MFCC.npy")

# Choose X
#X = MFCC
X = np.concatenate((MFCC, d_MFCC, dd_MFCC), axis=1)

# Put the right labels in a vector y
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
flokkarOlikir = np.array(['' for _ in range(200)], dtype=object)
flokkarOlikir[0:100] = "metal"
flokkarOlikir[100:200] = "country"
X2 = np.concatenate((MFCC[600:700], MFCC[200:300]), axis=0)


y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


# Split into training, testing and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150, random_state=42)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=150, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X, y, test_size=30, random_state=42)
#X2_train, X2_val, y2_train, y2_val = train_test_split(X_train, y_train, test_size=30, random_state=42)


# Scale the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)
scaler.fit(X2_train)
X2_train_scaled = scaler.transform(X2_train)
X2_test_scaled = scaler.transform(X2_test)
X2_val_scaled = scaler.transform(X2_val)

X2_embedded = TSNE(n_components=2).fit_transform(X2)
X_embedded = TSNE(n_components=2).fit_transform(X)
ylab = np.zeros(1000)
for i in range(10):
    ylab[i*100:(i+1)*100] = i
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=ylab)


# Saving the pic
fig = plt.figure()
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=ylab)
plt.title("TSNE dimension reduction")
plt.savefig("tsne.png")


