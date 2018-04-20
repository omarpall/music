import numpy as np
import matplotlib.pyplot as plt
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
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


# =============================================================================
# Skiptum gögnunum í þjálfunar- og prófunargögn
# =============================================================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=30)


# =============================================================================
# Skölum gögnin
# =============================================================================
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#----------------- Classifiers ----------------------

# =============================================================================
# Stillum flokkara og spáum
# =============================================================================
clf = LinearSVC(C=0.01)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)


# =============================================================================
# Finnum 3 líklegustu flokkana fyrir hvert lag
# =============================================================================
dec = clf.decision_function(X_test_scaled)
mostLikelyClasses = []
for song in range(dec.shape[0]):
    sortedSong = dec[song].argsort()[-3:][::-1]
    mostLikelyClasses.append(sortedSong)

# =============================================================================
# Búum til y fylki af tölum í stað strengja til að geta borið saman við 
# 3 líklegustu flokkana
# =============================================================================
X_test_len = len(X_test)
y_test_index = np.zeros(X_test_len)
for i in range(X_test_len):
    for j in range(10):
        if(y_test[i] == flokkar[j]):
            y_test_index[i] = j
            break

# =============================================================================
# Finnum Nákvæmni fyrir 3 líklegustu flokkana, 2 líklegustu flokkanaa og 
# líklegasta flokkinn
# =============================================================================
firstPlace = 0
secondPlace = 0
thirdPlace = 0
for i in range(X_test_len):
    if y_test_index[i] == mostLikelyClasses[i][0]:
        firstPlace = firstPlace + 1
    elif y_test_index[i] == mostLikelyClasses[i][1]:
        secondPlace = secondPlace + 1
    elif y_test_index[i] == mostLikelyClasses[i][2]:
        thirdPlace = thirdPlace + 1

firstPlaceAcc = (firstPlace/X_test_len)
secondPlaceAcc = (firstPlaceAcc + secondPlace/X_test_len)
thirdPlaceAcc = (secondPlaceAcc + thirdPlace/X_test_len)
print("Fyrsta nákvæmni", firstPlaceAcc, "Önnur nákvæmni", secondPlaceAcc, 
      "Þriðja nákvæmni", thirdPlaceAcc)
        
# =============================================================================
# # Confusion matrix
# =============================================================================
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
plt.savefig("LinearSVC.png")
acc = np.sum(y_pred==y_test)/len(y_pred)
print("The accuracy is:",acc)


    

