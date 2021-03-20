import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


votes=pd.read_csv('eggs6.csv' , sep=',')
#votes.info()



X=votes.drop(["t" , "votes"] , axis=1)


t=votes["t"]
y=votes["votes"]

sc=StandardScaler()
X=sc.fit_transform(X)

LogRes=LogisticRegression(penalty='none' , solver="lbfgs")
LogRes.fit(X,t)

"""sample_prop_scores=np.array(LogRes.predict_proba(X)[:,1])

ATE=(sum(y*t/sample_prop_scores)-sum(y*(1-t)/(1-sample_prop_scores)))/1215
print(ATE)"""


ATEs=[]
for i in range(1500):
    sample=votes.sample(1200,replace=True)
    X_sample=sample.drop(["t","votes"] , axis=1)
    t_sample=sample["t"]
    y_sample=sample["votes"]
    X_sample=sc.fit_transform(X_sample)
    
    sample_prop_scores=np.array(LogRes.predict_proba(X_sample)[:,1])
    ATE=(sum((y_sample*t_sample)/sample_prop_scores)-sum((y_sample*(1-t_sample))/(1-sample_prop_scores)))/1200
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
