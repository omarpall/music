import numpy as np


# Read the training data (8000 songs partitions, 3874 mfcc features each)
X_train = np.load("XTrainButar.npy")
y_train = np.load("yTrainButar.npy")
# Read the testing data ()
X_test = np.load("XTestButar.npy")
y_test = np.load("yTestButar.npy")





def predBit(fit, X_test):
    flokkar = ["blues","classical","country","disco","hiphop",
               "jazz","metal","pop","reggae","rock"]
    n,p = X_test.shape
    y_pred = np.array(['' for _ in range(n)], dtype=object)
    i=0
    for song in X_test:
        lag = song.reshape(10,-1)
        lagPred = np.zeros(len(flokkar))
        lagFlokkar = fit.predict(lag)
        for prediction in lagFlokkar:
            hvar = flokkar.index(prediction)
            lagPred[hvar] = lagPred[hvar]+1
        # prediction = np.argmax(lagPred)
        prediction = np.random.choice(np.flatnonzero(lagPred == np.max(lagPred)))
        y_pred[i] = flokkar[prediction]
        i += 1
    return(y_pred)


# Traina linearSVC
from sklearn.svm import LinearSVC
C = [1e-07,5e-07,1e-06,5e-06,1e-05]
best_C = 0
bestAcc = 0
for c in C:
    print("Training for C =",c)
    clf = LinearSVC(random_state=0,C=c)
    clf.fit(X_train, y_train)
    ypred = predBit(clf, X_test)
    acc = sum(ypred==y_test)/len(y_test)
    print(acc)
    if(acc>bestAcc):
        best_C=c
        bestAcc=acc

    
#svc = SVC(C=0.01, kernel='linear')
#svc.fit(X_train, y_train)

#ypred = predBit(svc, X_test)