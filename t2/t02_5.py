import numpy as np
from t02_1 import preprocess_data #import code from problem 1
from sklearn.decomposition import PCA

def apply_pca():
  #preprocess data
  train_data, train_labels, test_data, test_labels = preprocess_data()
  
  #fit PCA with train data
  model = PCA(n_components = 0.9).fit(train_data)
  
  #apply PCA to train/test data
  train_data = model.transform(train_data)
  test_data = model.transform(test_data)
  
  return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":
  train_data, train_labels, test_data, test_labels = apply_pca()
  
  #show data  
  print(train_data)
  print(train_labels)
  print(test_data)
  print(test_labels)

