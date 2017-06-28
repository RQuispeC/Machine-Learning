import numpy as np
import csv
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn import svm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def preprocess_data():
  #read cvs file
  file_obj = open("abalone.csv", "rt")
  reader = csv.reader(file_obj)
  data = []
  for row in reader:
    data.append(row)

  data = np.array(data)

  #extratct first column and convert data to floats
  first_column = [row[0] for row in data]
  data = np.array([row[1::] for row in   data]).astype(np.float)

  #convert first column using one-hot-encoding and restore it to data
  first_column = np.array(pd.get_dummies(first_column))
  data = np.hstack((first_column, data))

  #separe and transform last column
  labels = data[:, -1]
  labels[labels <= 13] = 0
  labels[labels > 13] = 1
  data = data[:, 0:-1]

  #standardize data
  scaler = preprocessing.StandardScaler().fit(data)
  data = scaler.transform(data)

  return data, labels

def logistic_regressor_evaluation(x, y):
  C_space = [10**i for i in range(-1, 4) ]

  #create stratified k-folds
  skFolds = model_selection.StratifiedKFold(n_splits = 5, random_state = 1)
  skFolds.get_n_splits(x, y)

  accuracy_mean = 0

  for train_index, test_index in skFolds.split(x, y):

    #set train and test data
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #create stratified k-folds for nested loop
    nested_skFolds = model_selection.StratifiedKFold(n_splits = 3, random_state = 1)
    nested_skFolds.get_n_splits(x_train, y_train)

    best_c = 0 
    best_accuracy = 0
    
    for c in C_space:
    
      c_accuracy = 0.0
      
      for nested_train_index, nested_test_index in nested_skFolds.split(x_train, y_train):

        #set nested train and test data
        nested_x_train, nested_x_test = x_train[nested_train_index], x_train[nested_test_index]
        nested_y_train, nested_y_test = y_train[nested_train_index], y_train[nested_test_index]
          

        #create and train the nested logistic Regressor
        nested_clf = LogisticRegression(C=c, random_state=1)
        nested_clf.fit(nested_x_train, nested_y_train)

        #test the trained model
        nested_pred = nested_clf.predict(nested_x_test)
        nested_accuracy = metrics.accuracy_score(nested_y_test, nested_pred)
       
        c_accuracy += nested_accuracy
        
      c_accuracy /= 3.0
      
      print("\tnested accuracy", c_accuracy, c)
      
      #update best accuracy
      if best_accuracy <= c_accuracy:
        best_accuracy =  c_accuracy
        best_c = c
    
    #train logistic Regressor with the best C
    clf = LogisticRegression(C=best_c, random_state=1)
    clf.fit(x_train, y_train)

    #test the trained model
    pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
       
    print("LR KF Accuracy", accuracy, best_c)

    #acumulate accuracy
    accuracy_mean += accuracy
  
  print("Logistic regressor mean accuracy", accuracy_mean/5.0)

def SVM_evaluation(x, y):
  parameters = {'C' : [10**i for i in range(-1, 4) ]}

  #create stratified k-folds
  skFolds = model_selection.StratifiedKFold(n_splits = 5, random_state = 1)
  skFolds.get_n_splits(x, y)

  accuracy_mean = 0

  for train_index, test_index in skFolds.split(x, y):

    #set train and test data
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #create nested SVM
    nested_svm = svm.LinearSVC(random_state = 1)

    #create nested loop and get best parametes
    nested_clf = GridSearchCV(nested_svm, parameters, cv = 3)
    nested_clf.fit(x_train, y_train)

    print("\tNested Accuracy", nested_clf.best_score_, nested_clf.best_params_)
    
    #train SVM with the best parameters
    clf = svm.LinearSVC(C = nested_clf.best_params_['C'], random_state = 1)
    clf.fit(x_train, y_train)

    #test the trained model
    pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
       
    print("SVM KF Accuracy", accuracy, nested_clf.best_params_['C'])

    #acumulate accuracy
    accuracy_mean += accuracy
  
  print("SVM mean accuracy", accuracy_mean/5.0)

def LDA_evaluation(x, y):
  skFolds = model_selection.StratifiedKFold(n_splits = 5, random_state = 1)
  skFolds.get_n_splits(x, y)

  accuracy_mean = 0

  for train_index, test_index in skFolds.split(x, y):

    #set train and test data
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #create and train LDA
    clf = LinearDiscriminantAnalysis()
    clf.fit(x_train, y_train)

    #test the trained model
    pred = clf.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, pred)
       
    print("LDA KF Accuracy", accuracy)

    #acumulate accuracy
    accuracy_mean += accuracy
  
  print("LDA mean accuracy", accuracy_mean/5.0)

def train_best_classifier(x, y):
  C_space = [10**i for i in range(-1, 4) ]

  #create stratified k-folds
  skFolds = model_selection.StratifiedKFold(n_splits = 3, random_state = 1)
  skFolds.get_n_splits(x, y)

  best_c = 0
  best_accuracy = 0
  
  for c in C_space:
    accuracy_mean = 0
    
    #test each parameter C with the 3-folds
    for train_index, test_index in skFolds.split(x, y):
      x_train, x_test = x[train_index], x[test_index]
      y_train, y_test = y[train_index], y[test_index]
  
      svc = svm.LinearSVC(random_state = 1, C=c).fit(x_train, y_train)
      pred = svc.predict(x_test)
      
      accuracy = metrics.accuracy_score(y_test, pred)
      accuracy_mean += accuracy
        
    accuracy_mean /= 3.0
    
    print("\t nested", accuracy_mean, c)
    
    #update best accuracy and best C
    if best_accuracy <= accuracy_mean:
      best_accuracy = accuracy_mean
      best_c = c
  
  print("best", best_accuracy, best_c)

  #create final classifier
  return svm.LinearSVC(C=best_c, random_state = 1).fit(x, y)

if __name__ == "__main__":
  x, y = preprocess_data()
 
  logistic_regressor_evaluation(x, y)

  SVM_evaluation(x, y)

  LDA_evaluation(x,y)

  train_best_classifier(x , y)
