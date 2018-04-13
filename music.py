import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# Dæmi um lestur fyrsta blues lagsins, og plot
#(rate,sig) = wav.read("waves/blues.00000.wav")
#plt.plot(np.arange(sig.shape[0])/rate,sig)

# Lesum inn gögnin og setjum mfcc features í fylki
MFCC = np.load("MFCC.npy")
d_MFCC = np.load("d_MFCC.npy")
dd_MFCC = np.load("dd_MFCC.npy")

# Choose X
X = np.concatenate((MFCC, d_MFCC, dd_MFCC), axis=1)

# Put the right labels in a vector y
flokkar = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
y = np.array(['' for _ in range(1000)], dtype=object)
for i in range(10):
    y[100*i:100*(i+1)] = flokkar[i]


# Split into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scale the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#----------------- Classifiers ----------------------

# Support Vector Machine 

parameters = {'kernel':('linear','rbf'), 'C':[0.1,1,10,100],'gamma':[.1,.01,.001]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(xTrain_scaled, yTrain)
acc = clf.score(xTest_scaled,yTest)
print("The accuracy is:",acc)
        
# Confusion matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test,y_pred, labels=flokkar) 
print(cm1)

