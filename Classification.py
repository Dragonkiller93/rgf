from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFClassifier, FastRGFClassifier
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
for var in range(10):
    param = (10*10**((var+1)//4))+250*(var%4)
    clf = FastRGFClassifier(n_estimators= param)
    clf.fit(Xtrain,Ytrain)
    score = clf.score(Xtest,Ytest)
    print("Number of Trees: " + str(param) + ". Score: " + str(score))

