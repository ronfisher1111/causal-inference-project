import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LinearRegression

#estimate 95% confidenct interval for the s_learner ATE

data=pd.read_csv('s_learner_data.csv' , sep=',')


X=data.drop("votes" , axis=1)
y=data["votes"]


sc=StandardScaler()
X=sc.fit_transform(X)

reg = LinearRegression()
reg.fit(X,y)

"""sample_prop_scores=np.array(LogRes.predict_proba(X)[:,1])

ATE=(sum(y*t/sample_prop_scores)-sum(y*(1-t)/(1-sample_prop_scores)))/1215
print(ATE)"""

data2=pd.read_csv('s_learner_data_1.csv' , sep=',')


n=900
ATEs=[]
for i in range(1000):
    sample=data2.sample(n,replace=True)
    X_sample=sample.drop("votes" , axis=1)
    t_sample=sample["t"]
    y_sample=sample["votes"]
    
    X_sample=sc.fit_transform(X_sample)
    
    predictions = np.array(reg.predict(X_sample))
    
    ATE=-1*sum(2*(t_sample-0.5)*(y_sample-predictions))/n
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
