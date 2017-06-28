import numpy as np
import csv
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA

from sklearn import discriminant_analysis

def preprocess_data():
  #read cvs file
  file_obj = open("abalone.csv", "rt")
  reader = csv.reader(file_obj)
  data = []
  for row in reader:
    data.append(row)
  data = np.array(data)

  #extract first column and convert data to floats
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

def kNN_evaluation(x, y):
    #PCA
    x = PCA(n_components = 0.9).fit_transform(x)

    print "K Neastest Neighbors"
    params = {'n_neighbors' : [1, 5, 11, 15, 21, 25]}
    #create stratified k-folds
    kf = StratifiedKFold(n_splits = 5, random_state = 1)
    kf.get_n_splits(x, y)

    accuracy_mean = 0
    for train_index, test_index in kf.split(x, y):
        #set train and test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #run gridSearch
        gridSearch = GridSearchCV(KNeighborsClassifier(), params, cv = 3)
        gridSearch.fit(x_train, y_train)

        #create model with best params
        clf = KNeighborsClassifier(**gridSearch.best_params_).fit(x_train, y_train)

        #test created model
        nested_accuracy = clf.score(x_test, y_test)
        accuracy_mean += nested_accuracy
        print "\tn_neighbors: ",gridSearch.best_params_['n_neighbors']," nested acc:", nested_accuracy

    print "Mean acc:", round(accuracy_mean/5.0, 3)

def svm_evaluation(x, y):
    print "Support Vector Machine"
    params = {'C' : [2**(-5), 2**(0), 2**(5), 2**(10)], 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
    #create stratified k-folds
    kf = StratifiedKFold(n_splits = 5, random_state = 1)
    kf.get_n_splits(x, y)

    accuracy_mean = 0
    for train_index, test_index in kf.split(x, y):
        #set train and test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #run gridSearch
        gridSearch = GridSearchCV(SVC(random_state = 1), params, cv = 3)
        gridSearch.fit(x_train, y_train)

        #create model with best params
        clf = SVC(**gridSearch.best_params_).fit(x_train, y_train)

        #test created model
        nested_accuracy = clf.score(x_test, y_test)
        accuracy_mean += nested_accuracy
        print "\tC: ",gridSearch.best_params_['C'], "gamma: ",gridSearch.best_params_['gamma']," nested acc:", nested_accuracy

    print "Mean acc:", round(accuracy_mean/5.0, 3)

def mlp_evaluation(x, y):
    print "Multi Layer Perceptron"
    params = {'hidden_layer_sizes' : [(3,), (7, ), (10, ), (20, )]}
    #create stratified k-folds
    kf = StratifiedKFold(n_splits = 5, random_state = 1)
    kf.get_n_splits(x, y)

    accuracy_mean = 0
    for train_index, test_index in kf.split(x, y):
        #set train and test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #run gridSearch
        gridSearch = GridSearchCV(MLPClassifier(random_state  = 1), params, cv = 3)
        gridSearch.fit(x_train, y_train)

        #create model with best params
        clf = MLPClassifier(**gridSearch.best_params_).fit(x_train, y_train)

        #test created model
        nested_accuracy = clf.score(x_test, y_test)
        accuracy_mean += nested_accuracy
        print "\thidden_layer_sizes: ",gridSearch.best_params_['hidden_layer_sizes'], " nested acc:", nested_accuracy

    print "Mean acc:", round(accuracy_mean/5.0, 3)

def rf_evaluation(x, y):
    print "Random Forest"
    params = {'n_estimators' : [100, 200, 400, 800], 'max_features' : [2, 3, 5, 7]}
    #create stratified k-folds
    kf = StratifiedKFold(n_splits = 5, random_state = 1)
    kf.get_n_splits(x, y)

    accuracy_mean = 0
    for train_index, test_index in kf.split(x, y):
        #set train and test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #run gridSearch
        gridSearch = GridSearchCV(RandomForestClassifier(random_state  = 1), params, cv = 3)
        gridSearch.fit(x_train, y_train)

        #create model with best params
        clf = RandomForestClassifier(**gridSearch.best_params_).fit(x_train, y_train)

        #test created model
        nested_accuracy = clf.score(x_test, y_test)
        accuracy_mean += nested_accuracy
        print "\tn_estimators: ",gridSearch.best_params_['n_estimators'], "max_features: ",gridSearch.best_params_['max_features'], " nested acc:", nested_accuracy

    print "Mean acc:", round(accuracy_mean/5.0, 3)

def gb_evaluation(x, y):
    print "Gradient Boosting Machine"
    params = {'n_estimators' : [30, 70, 100], 'learning_rate' : [0.1, 0.05], 'max_depth': [5]}
    #create stratified k-folds
    kf = StratifiedKFold(n_splits = 5, random_state = 1)
    kf.get_n_splits(x, y)

    accuracy_mean = 0
    for train_index, test_index in kf.split(x, y):
        #set train and test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #run gridSearch
        gridSearch = GridSearchCV(GradientBoostingClassifier(random_state  = 1), params, cv = 3)
        gridSearch.fit(x_train, y_train)

        #create model with best params
        clf = GradientBoostingClassifier(**gridSearch.best_params_).fit(x_train, y_train)

        #test created model
        nested_accuracy = clf.score(x_test, y_test)
        accuracy_mean += nested_accuracy
        print "\tn_estimators: ",gridSearch.best_params_['n_estimators'], "learning_rate: ",gridSearch.best_params_['learning_rate'], "max_depth: ",gridSearch.best_params_['max_depth'], " nested acc:", nested_accuracy

    print "Mean acc:", round(accuracy_mean/5.0, 3)

def train_best_classifier(x, y):
    #params for gridSearch
    params = {'hidden_layer_sizes' : [(3,), (7, ), (10, ), (20, )]}

    gridSearch = GridSearchCV(MLPClassifier(random_state  = 1), params, cv = 3)
    gridSearch.fit(x, y)

    print gridSearch.best_params_
    print gridSearch.cv_results_['mean_test_score']
    #train final classifier
    return MLPClassifier(**gridSearch.best_params_).fit(x, y)

if __name__ == "__main__":
  x, y = preprocess_data()

  #kNN_evaluation(x, y)
  #svm_evaluation(x, y)
  #mlp_evaluation(x, y)
  #rf_evaluation(x, y)
  #gb_evaluation(x, y)

  train_best_classifier(x, y)
