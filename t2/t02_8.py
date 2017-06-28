import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

def preprocess_data_imputation():
  #read cvs file
  file_obj = open("abalone-missing.csv", "rt")
  reader = csv.reader(file_obj)
  data = []
  for row in reader:
    data.append(row)
 
  #convert data to numpy array
  data = np.array(data)
 
  #replace missing data with np.nan
  data[data == 'NA'] = np.nan
  
  #get first column and update data columns
  first_column = data[:, 0]
  data = data[:, 1::]
  
  #convert first column using one-hot-encoding and restore it inside data
  first_column = np.array(pd.get_dummies(first_column))
  data = np.hstack((first_column, data))  
  
  #separe and transform last column
  labels = data[:, -1].astype(np.float)
  labels[labels <= 13] = 0
  labels[labels > 13] = 1
  data = data[:, 0:-1]
  
  #separe data for training and testing
  train_data = data[0:3133 , :]
  train_labels = labels[0:3133]
  
  test_data = data[3133::, :]
  test_labels = labels[3133::]
  
  #fit imputer with train data
  imp = Imputer(missing_values = "NaN", strategy = "mean").fit(train_data)
  
  #impute missing train/test data
  train_data = imp.transform(train_data)
  test_data = imp.transform(test_data)
  
  return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":

  train_data, train_labels, test_data, test_labels = preprocess_data_imputation()
  
  #create and train the logistic Regressor
  clf = LogisticRegression(C=1000000, random_state=1)
  clf.fit(train_data, train_labels)

  #test the trainet model
  pred = clf.predict(test_data)
  
  #show prediction statistics
  print (metrics.classification_report(test_labels, pred))
  print ("Accuracy: {0:.3f}".format(metrics.accuracy_score(test_labels, pred)))
  
