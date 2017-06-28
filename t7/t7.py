import numpy as np
import csv

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
from sklearn.neighbors import NearestNeighbors

import pandas as pd
from matplotlib import pyplot as plt

def preprocess_data():
  #read cvs file
  file_obj = open("quakes.csv", "rt")
  reader = csv.reader(file_obj)
  data = []
  for row in reader:
    data.append(row)
  data = np.array(data).astype(np.float)

  #standardize data
  scaler = StandardScaler().fit(data)
  data = scaler.transform(data)

  return data

def kmeans_evaluation(data):
    print "K-means"
    for k in range(2,11):
        #k-means clustering model and fit
        kmeans_model = KMeans(n_clusters = k, random_state=1).fit(data)
        labels = kmeans_model.labels_
        print "\tClusters:", k
        print "\tSilhouette: %.2f" % silhouette_score(data, labels, metric='euclidean')
        print "\tcalinski Harabaz: %.2f" % calinski_harabaz_score(data, labels)

def hierarchical_evaluation(data):
    print "Hirerchical- Ward method"
    for k in range(2,11):
        #hierarchical-means clustering model and fit
        hierarchical = AgglomerativeClustering(n_clusters = k, linkage = "ward").fit(data)
        labels = hierarchical.labels_
        print "\tClusters:", k
        print "\tSilhouette: %.2f" % silhouette_score(data, labels, metric='euclidean')
        print "\tcalinski Harabaz: %.2f" % calinski_harabaz_score(data, labels)


def DBSCAN_evaluation(data):
    print "DBSCAN"
    eps_values = [0.6, 0.65, 0.7]
    for eps_value in eps_values:
        DBSCAN_model = DBSCAN(min_samples = 5, eps = eps_value).fit(data)
        labels = DBSCAN_model.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print "\tEPS:", eps_value
        print "\tClusters:", n_clusters
        print "\tSilhouette: %.2f" % silhouette_score(data, labels, metric='euclidean')
        print "\tcalinski Harabaz: %.2f" % calinski_harabaz_score(data, labels)

def plotHistogram(distances, save_file_name = ""):
    plt.figure()
    df = pd.DataFrame({'d': distances})
    ax = df.hist(bins = 20)
    plt.savefig(save_file_name)

def plotDistances(distances, save_file_name = ""):
    distances.sort()
    plt.figure()
    plt.plot(range(len(distances)), distances, 'g.')
    plt.savefig(save_file_name)

def compute_EPS(data):
    nn = NearestNeighbors(n_neighbors = 6).fit(data)
    distances, indices = nn.kneighbors(data)
    graph = nn.kneighbors_graph(data).toarray()
    print distances[:,1:].mean()
    print distances[:,1:].min()
    print distances[:,1:].max()
    plotHistogram(distances[:, 1:].ravel(), "full_his.png")
    plotDistances(distances[:, 1:].ravel(), "full_dis.png")
    for i in range(1, 6):
        print "===================="
        print i
        print distances[:, i].mean()
        print distances[:, i].min()
        print distances[:, i].max()
        plotHistogram(distances[:, i].ravel(), str(i) + "_his.png")
        plotDistances(distances[:, i].ravel(), str(i) + "_dis.png")

def remove_lat_lon(data):
    #remove latitute and longitute columns
    return np.hstack((data[:, 0].reshape(-1, 1), data[:, 3].reshape(-1, 1)))

if __name__ == '__main__':
    data = preprocess_data()
    kmeans_evaluation(data)
    kmeans_evaluation(remove_lat_lon(data))

    hierarchical_evaluation(data)
    compute_EPS(data)
    DBSCAN_evaluation(data)
