# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:34:06 2018

@author: notandi
"""
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor

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
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, flokkarOlikir, test_size=30, random_state=42)
#X2_train, X2_val, y2_train, y2_val = train_test_split(X_train, y_train, test_size=30, random_state=42)


# Scale the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_val_scaled = scaler.transform(X_val)
scaler.fit(X2_train)
X2_train_scaled = scaler.transform(X2_train)
X2_test_scaled = scaler.transform(X2_test)
#X2_val_scaled = scaler.transform(X2_val)

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X2_train_scaled, y2_train)
ypred = neigh.predict(X2_test_scaled) 
acc = np.sum(ypred==y2_test)/len(ypred)
print("The accuracy is:",acc)