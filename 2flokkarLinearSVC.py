# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 22:28:06 2018

@author: notandi
"""

import numpy as np
from sklearn.svm import LinearSVC



# =============================================================================
# Lesum inn gögnin
# =============================================================================
MFCC = np.load("MFCC.npy")
X = MFCC


# =============================================================================
# Stillum fylkið y þannig að y[i] innihaldi flokkinn sem passar við X[i] 
# =============================================================================
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y2 = np.array(['' for _ in range(200)], dtype=object)
y2[0:100] = "classical"
y2[100:200] = "metal"
y3 = np.array(['' for _ in range(200)], dtype=object)
y3[0:100] = "blues"
y3[100:200] = "rock"

X2 = np.concatenate((MFCC[100:200], MFCC[600:700]), axis=0)
X3 = np.concatenate((MFCC[0:100], MFCC[900:1000]), axis=0)



# =============================================================================
# Skiptum gögnunum í þjálfunar- og prófunargögn
# =============================================================================
from sklearn.model_selection import train_test_split

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=40, random_state=30)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=40, random_state=30)

# =============================================================================
# Skölum gögnin
# =============================================================================
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X2_train)
X2_train_scaled = scaler.transform(X2_train)
X2_test_scaled = scaler.transform(X2_test)
scaler.fit(X3_train)
X3_train_scaled = scaler.transform(X3_train)
X3_test_scaled = scaler.transform(X3_test)


#----------------- Classifiers ----------------------



# =============================================================================
# Prófum flokkara með mismunandi max_features og könnum mun í nákvæmni
# =============================================================================
clf = LinearSVC(C=0.01)
clf.fit(X2_train_scaled, y2_train)
y2_pred = clf.predict(X2_test_scaled)
clf.fit(X3_train_scaled, y3_train)
y3_pred = clf.predict(X3_test_scaled)



# =============================================================================
# # Confusion matrix
# =============================================================================
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y2_test,y2_pred, labels=flokkar) 
cm3 = confusion_matrix(y3_test,y3_pred, labels=flokkar) 
print(cm2)
print(cm3)

acc2 = np.sum(y2_pred==y2_test)/len(y2_pred)
acc3 = np.sum(y3_pred==y3_test)/len(y3_pred)
print("Nákvæmni fyrir ólíka flokka er:",acc2)
print("Nákvæmni fyrir líka flokka er:",acc3)
