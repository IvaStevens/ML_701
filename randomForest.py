from sklearn import cross_validation
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from numpy import argmax, average, array, insert
from sklearn.linear_model import LogisticRegression
import math
import csv
import numpy

eps = 1e-15
nFolds = 10

def logloss(p, y):
    return -math.log(max(min(p[y-1], 1 - eps), eps))

def binloss(p, y):
    return (argmax(p)+1) != y

def loadTrain():
    X = []
    y = []
    with open('train_data_only.csv', 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            values = [int(v) for v in row]
            X.append(values[0:92])
            y.append(values[93])
    X = array(X)
    y = array(y)
    return X,y

def loadTest():
    X = []
    with open('test_data_only.csv', 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            values = [int(v) for v in row]
            X.append(values[0:92])
    X = array(X)
    return X

def saveResult(res):
    n, f = res.shape
    temp = numpy.zeros((n,f+1))
    temp[:,1:] = res
    for i in range(0,n):
        temp[i,0] = i+1;

    with open('out2.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerows(temp)

def crossValidation(clf, XTrain, yTrain):
    folds = cross_validation.KFold(nTrain, nFolds, shuffle = True, random_state=0)
    
    errs1 = []
    errs2 = []
    for train, test in folds:
        X, W = XTrain[train], XTrain[test]
        y, z = yTrain[train], yTrain[test]
        
        clf.fit(X, y)
        res = clf.predict_proba(W)
        err1 = average(map(binloss, res, z))
        err2 = average(map(logloss, res, z))
        errs1.append(err1)
        errs2.append(err2)

        print 'fold = {0:2}, accuracy = {1:5f}, logloss = {2:5f}'.format(len(errs1), 1 - err1, err2)

    print 'avg.accuracy = {0:5f}, avg.logloss = {1:5f}'.format(1 -average(errs1), average(errs2))

XTrain, yTrain = loadTrain()
XTest = loadTest()

nTrain = len(XTrain)

#clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
rndForest = RandomForestClassifier(n_estimators=500, max_features=30, max_depth=None, min_samples_split=1, random_state=0, n_jobs=-1)
rndForestClb = CalibratedClassifierCV(rndForest, method='isotonic', cv=10)

mnb = MultinomialNB(alpha = 0.0, fit_prior = False)
mnbClb = CalibratedClassifierCV(mnb, method='sigmoid', cv=10)

#gbc = GradientBoostingClassifier(n_estimators = 500, learning_rate = 1.0, max_depth = 1, random_state = 1)
#lr = LogisticRegression(penalty='l2', dual=False, tol=0.000001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=500, multi_class='ovr', verbose=0)

crossValidation(rndForestClb, XTrain, yTrain)

# rndForestClb.fit(XTrain, yTrain);
# res = rndForestClb.predict_proba(XTest);
# saveResult(res)


##scores = cross_validation.cross_val_score(clf, XTrain, yTrain, cv=10)                            
##
##clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
##scores = cross_validation.cross_val_score(clf, XTrain, yTrain)
##print scores.mean()                             
##
##clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
##scores = cross_validation.cross_val_score(clf, XTrain, yTrain)
##print scores.mean()
