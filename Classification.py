from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFClassifier
import pandas as pd
import numpy

df = pd.read_excel(r'classdata.xlsx')
labels = df['Y']
data = df.drop(columns=['Y'])
data = data[1:]
labels = labels[1:].astype('int')


Xtrain = data.head(-3000)
Ytrain = labels.head(-3000)
Xtest = data.tail(3000)
Ytest = labels.tail(3000)
for leaf in range(10):
    l2term = 1/((10**(leaf//2)))*(0.5**(leaf%2))
    clf = RGFClassifier(l2 = l2term)
    clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    print("L2 regularization term: " + str(l2term) + ". Score: " + str(score))

