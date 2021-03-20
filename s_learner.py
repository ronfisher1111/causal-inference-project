import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LinearRegression



data=pd.read_csv('s_learner_data.csv' , sep=',')
#votes.info()





"""sns.countplot(votes["degree"])
plt.show()"""


X=data.drop("votes" , axis=1)
y=data["votes"]

sc=StandardScaler()
X=sc.fit_transform(X)

reg = LinearRegression()
reg.fit(X,y)


data2=pd.read_csv('s_learner_data_1.csv' , sep=',')
data2=sc.fit_transform(data2)

predictions = np.array(reg.predict(data2))
np.savetxt("s_predictions2.csv", predictions, delimiter=",")


coef = reg.coef_
print (coef)



#choose prop scores only between 0.06 and 0.885
