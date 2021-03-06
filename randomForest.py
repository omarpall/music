# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:11:01 2018

@author: notandi
"""
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Lesum inn gögnin
# =============================================================================
MFCC = np.load("MFCC.npy")
X = MFCC


# =============================================================================
# Stillum fylkið y þannig að y[i] innihaldi flokkinn sem passar við X[i] 
# =============================================================================
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


# =============================================================================
# Skiptum gögnunum í þjálfunar- og prófunargögn
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=42)


# =============================================================================
# Skölum gögnin
# =============================================================================
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =============================================================================
# Prófum flokkara með mismunandi max_features og könnum mun í nákvæmni
# =============================================================================
from sklearn.ensemble import RandomForestClassifier
for i in range(-3, 7):
    features = int(2**i * np.sqrt(len(X[0])))
    print(features)
    rf = RandomForestClassifier(n_estimators=100, n_jobs = -1, max_features=features, oob_score=True) # m=sqrt(n_features)
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    acc = np.average(y_test == y_pred)
    print("RF Test set accuracy for",features,"features: ", acc)



# =============================================================================
# Confusin matrix
# =============================================================================
rf = RandomForestClassifier(n_estimators=100, n_jobs = -1, max_features=3151, oob_score=True) 
rf.fit(X_train_scaled, y_train)
y_pred = rf.predict(X_test_scaled)
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred, labels=flokkar) 
print(cm1)
flokkaAcc = cm1/np.sum(cm1, axis = 1, keepdims = True)
print("Nákvæmni fyrir hvern flokk:")
print(np.round(np.diagonal(flokkaAcc*100),1))
plt.matshow(flokkaAcc, cmap="afmhot")
plt.xticks(range(len(flokkar)), flokkar, rotation=80)
plt.yticks(range(len(flokkar)), flokkar)
plt.colorbar()
plt.savefig("cmRandomForest.png")
acc = np.sum(y_pred==y_test)/len(y_pred)
print("The accuracy is:",acc)