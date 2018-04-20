# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:34:06 2018

@author: notandi
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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
# Prófum flokkara með fjölda nágranna frá 1 upp í 14
# =============================================================================
for i in range(1,15):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train_scaled, y_train)
    acc = neigh.score(X_test_scaled, y_test)
    print("For",i,"nearest neighbors we get an accuracy of ",acc)
