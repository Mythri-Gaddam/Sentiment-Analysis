import pandas as p
from sklearn.metrics import accuracy_score
import numpy as np
import sklearn
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.externals import joblib
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


# Classifiers

# SVM
svm_clf = LinearSVC()
pipeline_svm = make_pipeline(vect, svm_clf)
pipeline_svm.fit(train['text'].values.astype('U'),train['labels'])
prediction_svm = svm_clf.predict(y_train_count)

print("For the SVM classifier:")
print("Accuracy for SVM")
print(accuracy_score(test['labels'],prediction_svm))

# Saving the model
joblib.dump(svm_clf, 'final_model_SVM.sav')


# MultiLayer Perceptron
mlp_clf = MLPClassifier(alpha=1e-5,random_state=1,hidden_layer_sizes=(1000,),learning_rate='adaptive')
pipeline_mlp = make_pipeline(vect, mlp_clf)
pipeline_mlp.fit(train['text'].values.astype('U'),train['labels'])
prediction_mlp = mlp_clf.predict(y_train_count)

print("For the MLP classifier:")
print("For the accuracy of  MLP")
print(accuracy_score(test['labels'],prediction_mlp))

# Saving the model
joblib.dump(mlp_clf, 'final_model_MLP.sav')


# Random Forest
rf_clf = RandomForestClassifier(random_state=0)
pipeline_rf = make_pipeline(vect, rf_clf)
pipeline_rf.fit(train['text'].values.astype('U'),train['labels'])
prediction_rf = rf_clf.predict(y_train_count)

print("For the Random classifier:")
print("For the accuracy for RandomForest")
print(accuracy_score(test['labels'],prediction_rf))

# Saving the model
joblib.dump(rf_clf, 'final_model_RF.sav')
