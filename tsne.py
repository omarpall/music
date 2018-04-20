# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:58:02 2018
@author: notandi
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# =============================================================================
# Lesum inn gögnin, X inniheldur lög frá öllum flokkunum, X2 inniheldur aðeins
# lög frá metal og country(ólíkum flokkum) og X3 inniheldur lög frá blues og
# og rock(líkum flokkum).
# =============================================================================
MFCC = np.load("MFCC.npy")
X = MFCC
X2 = np.concatenate((MFCC[600:700], MFCC[200:300]), axis=0)
X3 = np.concatenate((MFCC[0:100], MFCC[900:1000]), axis=0)



# =============================================================================
# Prófum t-SNE klösun á gögnunum og plottum niðurstöðurnar
# =============================================================================
X_embedded = TSNE(n_components=2).fit_transform(X)
X2_embedded = TSNE(n_components=2).fit_transform(X2)
X3_embedded = TSNE(n_components=2).fit_transform(X3)

ylab = np.zeros(1000)
for i in range(10):
    ylab[i*100:(i+1)*100] = i
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=ylab, cmap="tab10")
plt.colorbar()
plt.xlim((-30,30))
plt.ylim((-20,20))
plt.title("Allir flokkar")
plt.savefig("tsne10.png")

y2lab = np.zeros(200)
for i in range(2):
    y2lab[i*100:(i+1)*100] = i
fig = plt.figure()
plt.scatter(X2_embedded[:,0], X2_embedded[:,1], c=y2lab)
plt.xlim((-50,50))
plt.ylim((-50,50))
plt.title("Tveir ólíkir flokkar")
plt.savefig("tsne2Olikir.png")

y3lab = np.zeros(200)
for i in range(2):
    y3lab[i*100:(i+1)*100] = i
fig = plt.figure()
plt.scatter(X3_embedded[:,0], X3_embedded[:,1], c=y3lab)
plt.title("Tveir líkir flokkar")
plt.savefig("tsne2Likir.png")


