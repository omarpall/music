import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
# Dæmi um lestur fyrsta blues lagsins, og plot
#(rate,sig) = wav.read("waves/blues.00000.wav")
#plt.plot(np.arange(sig.shape[0])/rate,sig)

# Read the data (mfcc features, deltas and delta-deltas)
MFCC = np.load("MFCC.npy")
d_MFCC = np.load("d_MFCC.npy")
dd_MFCC = np.load("dd_MFCC.npy")

# Choose X
X = MFCC
#X = np.concatenate((MFCC, d_MFCC, dd_MFCC), axis=1)

# Put the right labels in a vector y
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


# Split into training, testing and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=150, random_state=42)
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=150, random_state=42)


# Scale the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
#X_val_scaled = scaler.transform(X_val)

#----------------- Classifiers ----------------------

# Support Vector Machine - linear kernel
C = [.0000001,.000001,.00001,.0001,.001,.01,1]

# 100 random indexes from the train
rand = np.random.choice(len(X_train),size=700,replace=False)




clf = LinearSVC(random_state=0)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
print(clf.intercept_)
dec = clf.decision_function(X)
print(dec.shape)
print(dec[600:610])
        
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred, labels=flokkar) 
print(cm1)

