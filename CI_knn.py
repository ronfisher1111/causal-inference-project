import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

#estimate 95% confidenct interval for the s_learner ATE
neigh = KNeighborsRegressor(n_neighbors=5)
control=pd.read_csv('t_0.csv' , sep=',')
X=control.drop("votes" , axis=1)
y=control["votes"]

sc=StandardScaler()
X=sc.fit_transform(X)

neigh.fit(X, y)



treated=pd.read_csv('t_1.csv' , sep=',')


n=900
ATEs=[]
for i in range(1000):
    sample=treated.sample(n,replace=True)
    X_sample=sample.drop("votes" , axis=1)
    y_sample=sample["votes"]
    
    X_sample=sc.fit_transform(X_sample)
    
    predictions =neigh.predict(X_sample)
    
    ATE=sum(y_sample-predictions)/n
    ATEs.append(ATE)


boostrap=pd.DataFrame({'prop' : np.array(ATEs)})
print(boostrap.prop.quantile(0.025))
print(boostrap.prop.quantile(0.975))


"""#histogram
treat_plt = plt.hist(prop_scores[t==1], fc=(0, 0, 1, 0.5),bins=20,label='Treated')
cont_plt = plt.hist(prop_scores[t==0],fc=(1, 0, 0, 0.5),bins=20,label='Control')
plt.legend()
plt.xlabel('propensity score')
plt.ylabel('number of counties')
plt.show()





#choose prop scores only between 0.06 and 0.885"""
