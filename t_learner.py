import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LinearRegression



data_1=pd.read_csv('t_1.csv' , sep=',')
data_0=pd.read_csv('t_0.csv' , sep=',')



X_0=data_0.drop("votes" , axis=1)
X_1=data_1.drop("votes" , axis=1)
y_0=data_0["votes"]
y_1=data_1["votes"]

sc_0=StandardScaler()
X_0=sc_0.fit_transform(X_0)

sc_1=StandardScaler()
X_1=sc_1.fit_transform(X_1)

reg_0 = LinearRegression()
reg_0.fit(X_0,y_0)

reg_1 = LinearRegression()
reg_1.fit(X_1,y_1)



predictions = np.array(reg_0.predict(X_1))
np.savetxt("t_predictions_0.csv", predictions, delimiter=",")

predictions = np.array(reg_1.predict(X_0))
np.savetxt("t_predictions_1.csv", predictions, delimiter=",")


print (reg_0.coef_)
print (reg_1.coef_)



#choose prop scores only between 0.06 and 0.885
