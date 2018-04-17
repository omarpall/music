# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:11:01 2018

@author: notandi
"""
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.svm import SVC
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
y2flokkar = np.array(['' for _ in range(200)], dtype=object)
y2flokkar[0:100] = "blues"
y2flokkar[100:200] = "rock"
X2 = np.concatenate((X[0:100], X[900:1000]), axis=0)


y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


# Split into training, testing and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=300, random_state=42)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=150, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2flokkar, test_size=60, random_state=42)
#X2_train, X2_val, y2_train, y2_val = train_test_split(X2_train, y2_train, test_size=30, random_state=42)


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

# Random Forests flokkari
from sklearn.ensemble import RandomForestClassifier

# Þennan flokkara má (líkast til) útfæra með því að breyta einni línu í heimasmíðaða bagging kóðanum 
# hér að ofan: clf = tree.DecisionTreeClassifier(max_features=m,splitter='random')
# þar sem m<n_features er stiki, t.d. m=sqrt(n_features). Það sem þá gerist er 
# að m inntaksbreytur eru valdar af handahófi og besta skipting fundin m.t.t. þeirra
# (í stað þess að fara í gegnum allar breyturnar)

rf = RandomForestClassifier(n_estimators=100000, n_jobs = -1, max_features='sqrt', oob_score=True) # m=sqrt(n_features)
#Only using 2 categories
rf.fit(X2_train_scaled, y2_train)
y2_pred = rf.predict(X2_test_scaled)
acc = np.average(y2_test == y2_pred)
print("RF Test set accuracy: ", acc)

#Using all categories
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
acc = np.average(y_test == y_pred)
print("RF Test set accuracy: ", acc) # Bera saman við 0.08 sem fékkst fyrir eitt tré
#print("RF Out-of-bag error rate: ", 1-ens.oob_score_)