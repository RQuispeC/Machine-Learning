import numpy as np
from t02_5 import apply_pca #import code from problem 5
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

if __name__ == "__main__":
  #recover, preprocess and apply pca
  train_data, train_labels, test_data, test_labels = apply_pca()
  
  #create and train the logistic Regressor
  clf = LogisticRegression(C=1, random_state=1)
  clf.fit(train_data, train_labels)

  #test the trainet model
  pred = clf.predict(test_data)
  
  #show prediction statistics
  print (metrics.classification_report(test_labels, pred))
  print ("Accuracy: {0:.3f}".format(metrics.accuracy_score(test_labels, pred)))

