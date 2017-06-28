import numpy as np
import csv
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.decomposition import PCA
from sklearn.externals import joblib


from sklearn import discriminant_analysis
def read_data():
  file_name = "small.csv"
  print "WORWING ON", file_name
  #read cvs file
  file_obj = open(file_name, "rt")
  reader = csv.reader(file_obj)
  data = []
  for row in reader:
    data.append(row)
  data = np.array(data)
  if file_name == "ex6-train.csv":
    data = data[1::, :].astype(np.float) #remove headers
  else:
    data = np.array(data).astype(np.float)

  labels = data[:, -1]
  data = data[:, 0:-1]
  return data, labels

def preprocess_data(x, standardize = True, pca = True):
  #standardize data
  if standardize:
      x = preprocessing.StandardScaler().fit_transform(x)
  if pca:
      x = PCA(n_components =  0.9999).fit_transform(x)

  return x


def three_validation(x, y, clf_name, params, name_metada = "", cnt = 1):
    clfs = { 'knn': KNeighborsRegressor, 'svm': SVR, 'mlp': MLPRegressor, 'rf': RandomForestRegressor, 'gbm': GradientBoostingRegressor, 'brm': BaggingRegressor}
    gridSearch = GridSearchCV(clfs[clf_name](), params, cv = 3, scoring = 'neg_mean_absolute_error', n_jobs = 5)
    gridSearch.fit(x, y)
    name = clf_name + str(gridSearch.best_params_) + str(gridSearch.best_score_) + name_metada
    if clf_name != 'brm':
        joblib.dump(gridSearch, name + '_GS.pkl')
    else:
        print name
        joblib.dump(gridSearch, str(cnt) + name_metada + '_GS.pkl')
    clf_eval = clfs[clf_name](**gridSearch.best_params_).fit(x, y)
    if clf_name != 'brm':
        joblib.dump(clf_eval, name + '_REG.pkl')
    else:
        joblib.dump(clf_eval, str(cnt) + name_metada + '_REG.pkl')


def five_three_validation(x, y, clf_name, params):
    clfs = { 'knn': KNeighborsRegressor, 'svm': SVR, 'mlp': MLPRegressor, 'rf': RandomForestRegressor, 'gbm': GradientBoostingRegressor, 'brm': BaggingRegressor}
    kf = KFold(n_splits = 5, random_state = 1)
    kf.get_n_splits(x, y)

    accuracy_mean = 0
    for train_index, test_index in kf.split(x, y):
        #set train and test data
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #run gridSearch
        gridSearch = GridSearchCV(clfs[clf_name](), params, cv = 3, scoring = 'neg_mean_absolute_error', n_jobs = 5)
        gridSearch.fit(x_train, y_train)

        #create model with best params
        clf_eval = clfs[clf_name](**gridSearch.best_params_).fit(x_train, y_train)

        #test created model
        pred = clf_eval.predict(x_test)
        nested_accuracy = mean_absolute_error(y_test, pred)
        accuracy_mean += nested_accuracy
        print "\t", gridSearch.best_params_, "nested mae:", round(nested_accuracy, 10)

    print "Mean mae:", round(accuracy_mean/5.0, 10)

def kNN_evaluation(x, y, model = "five_three", name_metada = ""):
    print "K Neastest Neighbors"
    params = {'n_neighbors' : [1, 5, 11, 15, 21, 25]}
    if model == "five_three":
        five_three_validation(x, y, 'knn', params)
    if model == "three":
        three_validation(x, y, 'knn', params, name_metada)

def svm_evaluation(x, y, model = "five_three", name_metada = ""):
    print "Support Vector Machine"
    params = {'kernel': ['linear', 'rbf'], 'C' : [2**(-5), 2**(0), 2**(5), 2**(10)], 'gamma': [2**(-15), 2**(-10), 2**(-5), 2**(0), 2**(5)]}
    if model == "five_three":
        five_three_validation(x, y, 'svm', params)
    if model == "three":
        three_validation(x, y, 'svm', params, name_metada)

def mlp_evaluation(x, y, model = "five_three", name_metada = ""):
    print "Multi Layer Perceptron"
    params = {'hidden_layer_sizes' : [(3,), (7, ), (10, ), (20, )], 'solver' : ['lbfgs']}
    if model == "five_three":
        five_three_validation(x, y, 'mlp', params)
    if model == "three":
        three_validation(x, y, 'mlp', params, name_metada)
def rf_evaluation(x, y, model = "five_three", name_metada = ""):
    print "Random Forest"
    #params = {'n_estimators' : [100, 200, 400, 800], 'max_features' : [2, 3, 5, 7]}
    params = {'n_estimators' : [100, 200, 400, 800], 'max_features' : [8, 10, 30, 32]}
    if model == "five_three":
        five_three_validation(x, y, 'rf' , params)
    if model == "three":
        three_validation(x, y, 'rf' , params, name_metada)

def gb_evaluation(x, y, model = "five_three", name_metada = ""):
    print "Gradient Boosting Machine"
    params = {'n_estimators' : [30, 70, 100, 400, 800], 'learning_rate' : [0.1, 0.05], 'max_depth': [5, 10]}
    if model == "five_three":
        five_three_validation(x, y, 'gbm', params)
    if model == "three":
        three_validation(x, y, 'gbm', params, name_metada)

def brm_evaluation(x, y, model_evaluation = "five_three", name_metada = ""):
    '''
    models = [MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = (3,)),
              MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = (7,)),
              MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = (10,)),
              MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = (20,)),
              KNeighborsRegressor(n_neighbors = 1),
              KNeighborsRegressor(n_neighbors = 5),
              KNeighborsRegressor(n_neighbors = 10),
              KNeighborsRegressor(n_neighbors = 15)]
    '''
    models = [MLPRegressor(solver = 'lbfgs', hidden_layer_sizes = (20,))]
    cnt = 666
    for model in models:
        print "Bagging Regressor Machine"
        #params = {'base_estimator': [model], 'n_estimators' : [100, 200, 400, 800], 'max_features' : [5, 7, 8, 10, 30, 32, 34, 40]}
        params = {'base_estimator': [model], 'n_estimators' : [100, 400, 800], 'max_features' : [5,10, 30,40]}
        if model_evaluation == "five_three":
            five_three_validation(x, y, 'brm', params)
        if model_evaluation == "three":
            three_validation(x, y, 'brm', params, name_metada, cnt)
        cnt += 1

def evaluate_finalRegressor():
    reg = joblib.load('models/big/666_1_0_REG.pkl') 
    data = pd.read_csv('ex6-test.csv')
    data = np.array(data).astype(np.float)
    print  data.shape
    pred = reg.predict(data)
    df = pd.DataFrame(pred)
    df.to_csv("solutions.csv", header = False, index = False)

if __name__ == "__main__":
    evaluate_finalRegressor()
    x, y = read_data()
    x = preprocess_data(x, standardize = False, pca = False)
    #print "standardize = False, pca = False ======================== "
    #kNN_evaluation(x, y)
    #svm_evaluation(x, y)
    #mlp_evaluation(x, y)
    #rf_evaluation(x, y)
    #gb_evaluation(x, y)

    #xx = preprocess_data(x.copy(), standardize = True, pca = False)
    #print "standardize = True, pca = False ======================== "
    #kNN_evaluation(xx, y)
    #svm_evaluation(xx, y)
    #mlp_evaluation(xx, y)
    #rf_evaluation(xx, y)
    #gb_evaluation(xx, y)

    #xxx = preprocess_data(x.copy(), standardize = False, pca = True)
    #print "standardize = False, pca = True ======================== "
    #kNN_evaluation(xxx, y)
    #svm_evaluation(xxx, y)
    #mlp_evaluation(xxx, y)
    #rf_evaluation(xxx, y)
    #gb_evaluation(xxx, y)


    print "standardize = False, pca = False ======================== "
    #kNN_evaluation(x, y, "three", name_metada = "_0_0")
    #mlp_evaluation(x, y, "three", name_metada = "_0_0")
    #svm_evaluation(x, y, "three", name_metada = "_0_0")
    #rf_evaluation(x, y, "three", name_metada = "_0_0")
    #brm_evaluation(x, y,"three", name_metada = "_0_0")
    #gb_evaluation(x, y, "three", name_metada = "_0_0")

    xx = preprocess_data(x.copy(), standardize = True, pca = False)
    print "standardize = True, pca = False ======================== "
    #kNN_evaluation(xx, y, "three", name_metada = "_1_0")
    #svm_evaluation(xx, y, "three", name_metada = "_1_0")
    #mlp_evaluation(xx, y, "three", name_metada = "_1_0")
    #rf_evaluation(xx, y, "three", name_metada = "_1_0")
    #brm_evaluation(xx, y,"three", name_metada = "_1_0")
    #gb_evaluation(xx, y, "three", name_metada = "_1_0")

    xxx = preprocess_data(x.copy(), standardize = False, pca = True)
    print "standardize = False, pca = True ======================== "
    #kNN_evaluation(xxx, y, "three", name_metada = "_0_1")
    #svm_evaluation(xxx, y, "three", name_metada = "_0_1")
    #mlp_evaluation(xxx, y, "three", name_metada = "_0_1")
    #rf_evaluation(xxx, y, "three", name_metada = "_0_1")
    #brm_evaluation(xxx, y,"three", name_metada = "_0_1")
    #gb_evaluation(xxx, y, "three", name_metada = "_0_1")
    
    
