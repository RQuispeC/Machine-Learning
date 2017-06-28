import csv
import numpy as np
import pandas as pd


#read cvs file
file_obj = open("ex6-train.csv", "rt")
reader = csv.reader(file_obj)
data = []
for row in reader:
    data.append(row)
data = np.array(data)
np.random.shuffle(data)

data = data[:7000, :]
df = pd.DataFrame(data)
df.to_csv("small_7000.csv", header = False, index = False)
