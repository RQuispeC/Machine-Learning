import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
#from skelarn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

import pylab as plt

def read_data():
  #read cvs file
  data = pd.read_csv('data8.csv')
  data = np.array(data).astype(np.float)

  return data

def test_algorithm(data, alg_name, params):
    print 'Testing', alg_name, 'with', params
    alg = {'svm': OneClassSVM, 'isolation_forest': IsolationForest}
    clf = alg[alg_name](**params).fit(data)
    pred = clf.predict(data)
    for i in range(len(pred)):
        if(pred[i] == -1):
            print "Outlier:", i

def evaluate_SVM(data):
    for gamma in [2**(-85), 2**(-70), 2**(-50), 2**(-25), 2**(-15), 2**(-5)]:
        params = {'kernel': 'rbf', 'gamma': gamma}
        test_algorithm(data, 'svm', params)

def evaluate_IF(data):
    for n_estimators in [100, 300, 500, 700]:
        params = {'n_estimators' : n_estimators, 'contamination': 0.0077177}
        test_algorithm(data, 'isolation_forest', params)
'''
def evaluate_LOF(data):
    for n_estimators in [100, 300, 500, 700]:
        params = {'n_neighbors' : n_neighbors}
        test_algorithm(data, 'local_outlier_factor', params)
'''
def DBSCAN_evaluation(data):
    print "DBSCAN"
    eps_values = [0.6, 0.65, 0.7]
    for eps_value in eps_values:
        DBSCAN_model = DBSCAN(min_samples = 7, eps = eps_value).fit(data)
        labels = DBSCAN_model.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print "\tEPS:", eps_value
        print "\tClusters:", n_clusters
        print "\tSilhouette: %.2f" % silhouette_score(data, labels, metric='euclidean')
        print "\tcalinski Harabaz: %.2f" % calinski_harabaz_score(data, labels)

def plotDistances(distances, save_file_name = ""):
    distances.sort()
    plt.figure()
    plt.plot(range(len(distances)), distances, 'g.')
    plt. axhline (y=1 , color='#EE1212' )
    plt.savefig(save_file_name)

def nneighbors(data):
    nn = NearestNeighbors(n_neighbors = 9).fit(data)
    distances, indices = nn.kneighbors(data)

    for i in range(distances.shape[0]):
        distance = distances[i]
        indice = indices[i]
        if distance[8] >= 1:
            print "OUTLIER", i, indice[8], distance[8]

    plotDistances(distances[:, 8].ravel(), "dis.png")

if __name__ == '__main__':
    data = read_data()

    #nneighbors(data)
    evaluate_IF(data)
