from sklearn import datasets
from sklearn.utils.validation import check_random_state
from sklearn.model_selection import StratifiedKFold, cross_val_score
from rgf.sklearn import RGFClassifier
import pandas as pd

df = pd.read_excel(r'classdata.xlsx')
labels = df['Y']
data = df.drop(columns=['Y'])
data = data[1:]
labels = labels[1:].astype('int')
print(labels)
clf = RGFClassifier()
clf.fit(data,labels)
score = clf.score(data,labels)
print(score)
