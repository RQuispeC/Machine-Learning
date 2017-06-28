import numpy as np
import csv
import pandas as pd

def preprocess_data():
  #read cvs file
  file_obj = open("abalone.csv", "rt")
  reader = csv.reader(file_obj)
  data = []
  for row in reader:
    data.append(row)
      
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
  
  #separe data for training and testing
  train_data = data[0:3133 , :]
  train_labels = labels[0:3133]
  
  test_data = data[3133::, :]
  test_labels = labels[3133::]
  
  return train_data, train_labels, test_data, test_labels

if __name__ == "__main__":

  train_data, train_labels, test_data, test_labels = preprocess_data()
  
  print(len(train_data))
  print(train_labels)
  print(len(test_data))
  print(test_labels)
  
