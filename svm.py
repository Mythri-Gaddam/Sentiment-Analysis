import pandas as p
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import nltk
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Reading the file
train_file = p.read_csv('reviews.csv',sep='|',names=['labels','text'],encoding = "ISO-8859-1")
train_file=train_file.drop([0])
lines=len(train_file)
index = list(range(5,lines,5))
test = train_file.ix[index]
train = train_file.drop(index)


# Vectorizing the input

vect = CountVectorizer(stop_words='english')
vect2= vect
x_train_count = vect.fit_transform(train['text'].values.astype('U'))
vocab = vect.get_feature_names()
y_train_count = vect2.transform(test['text'].values.astype('U'))


# Classifier

classifier_rbf = LinearSVC()
pipeline1 = make_pipeline(vect, classifier_rbf)
pipeline1.fit(train['text'].values.astype('U'),train['labels'])
prediction_rbf = classifier_rbf.predict(y_train_count)
print("For the SVM classifier:")
print("Accuracy for SVM")
print(accuracy_score(test['labels'],prediction_rbf))
joblib.dump(classifier_rbf, 'final_model_SVM.sav')


# Saving the model
joblib.dump(pipeline1,'svm.mi')
