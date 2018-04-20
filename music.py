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
#X = np.concatenate((MFCC, d_MFCC), axis=1)
X = np.concatenate((MFCC, d_MFCC, dd_MFCC), axis=1)

# Put the right labels in a vector y
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]

# Split into training, testing and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=200, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=150, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# X_val_scaled = scaler.transform(X_val)

#----------------- Classifiers ----------------------

# ------- Support Vector Machine -------
# - Linear kernel

        
# SVC Cross-validation Grid Search - linear
parameters = {'C':[0.01,0.1,1,10,100]}
svc = SVC(kernel='linear')
clfLin = GridSearchCV(svc, parameters,n_jobs=-1)
clfLin.fit(X_train_scaled, y_train)
accTrain = clfLin.score(X_train_scaled,y_train)
accTest = clfLin.score(X_test_scaled,y_test)
print("The training accuracy is:",accTrain)
print("The testing accuracy is:",accTest)
print("The best parameters are:",clfLin.best_params_)

# SVC Cross-validation Grid Search - polynomial
parameters = {'C':[0.01,0.1,1,10,100],'degree':[2,3,4]}
svc = SVC(kernel='poly')
clfPol = GridSearchCV(svc, parameters,n_jobs=-1)
clfPol.fit(X_train_scaled, y_train)
accTrain = clfPol.score(X_train_scaled,y_train)
accTest = clfPol.score(X_test_scaled,y_test)
print("The training accuracy is:",accTrain)
print("The testing accuracy is:",accTest)
print("The best parameters are:",clfPol.best_params_)

# SVC Cross-validation Grid Search - rbf
parameters = {'C':[0.1,1,10,100],'gamma':[10,1,.1,.01,.001,.0001]}
svc = SVC(kernel='rbf')
clfRbf = GridSearchCV(svc, parameters,n_jobs=-1)
clfRbf.fit(X_train_scaled, y_train)
accTrain = clfRbf.score(X_train_scaled,y_train)
accTest = clfRbf.score(X_test_scaled,y_test)
print("The training accuracy is:",accTrain)
print("The testing accuracy is:",accTest)
print("The best parameters are:",clfRbf.best_params_)



bestSvc = SVC(C=0.1, kernel='linear')
bestSvc.fit(X_train_scaled, y_train)
acc = bestSvc.score(X_test_scaled, y_test)
print("We get a training accuracy of",acc)

y_pred = bestSvc.predict(X_test_scaled)
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred, labels=flokkar) 
print(cm1)




