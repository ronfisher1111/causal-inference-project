import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LinearRegression

#estimate 95% confidenct interval for the T_learner ATE



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



data=pd.read_csv('t.csv' , sep=',')


n=900
ATEs=[]
for i in range(1000):
    sample=data.sample(n,replace=True)

    grouped =sample.groupby(["t"])



    t_1 = grouped.get_group(1)
    t_0 = grouped.get_group(0)
    
    X_1_sample=t_1.drop(["t" ,"votes"] , axis=1)
    X_0_sample=t_0.drop(["t" ,"votes"] , axis=1)
    

    y_1_sample=t_1["votes"]
    y_0_sample=t_0["votes"]
    
    X_1_sample=sc_1.fit_transform(X_1_sample)
    X_0_sample=sc_0.fit_transform(X_0_sample)
    
    predictions_0 = np.array(reg_0.predict(X_1_sample))
    predictions_1 = np.array(reg_1.predict(X_0_sample))
    
    ATE=(sum(y_1_sample-predictions_0)-sum(y_0_sample-predictions_1))/n

    
    ATEs.append(ATE)


boostrap=pd.DataFrame({'prop' : np.array(ATEs)})
print(boostrap.prop.quantile(0.025))
print(boostrap.prop.quantile(0.975))



