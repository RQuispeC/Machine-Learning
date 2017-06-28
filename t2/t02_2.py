import numpy as np
from t02_1 import preprocess_data #import code from problem 1
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == "__main__":
  train_data, train_labels, test_data, test_labels = preprocess_data()
  
  #create and train the logistic Regressor
  clf = LogisticRegression(C=1000000, random_state=1)
  clf.fit(train_data, train_labels)

  #test the trainet model
  pred = clf.predict(test_data)
  
  #show prediction statistics
  print (metrics.classification_report(test_labels, pred))
  print ("Accuracy: {0:.3f}".format(metrics.accuracy_score(test_labels, pred)))

  
