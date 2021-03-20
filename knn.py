import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

#estimate ATT using KNN method

neigh = KNeighborsRegressor(n_neighbors=5)

control=pd.read_csv('t_0.csv' , sep=',')
X=control.drop("votes" , axis=1)
y=control["votes"]

sc=StandardScaler()
X=sc.fit_transform(X)

neigh.fit(X, y)

treated=pd.read_csv('t_1.csv' , sep=',')
treated_X=treated.drop("votes" , axis=1)

treated_X=sc.fit_transform(treated_X)
pred=neigh.predict(treated_X)
np.savetxt("knn_results.csv", pred, delimiter=",")
