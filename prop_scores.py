import pandas as pd
import seaborn as sns
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve


votes=pd.read_csv('data.csv' , sep=',')
#votes.info()

bins=(0,14.6 , 100)
group_names=["control" , "treated"]

votes["degree"]=pd.cut(votes["degree"] , bins=bins , labels=group_names)
label_quality=LabelEncoder()

votes["degree"]=label_quality.fit_transform(votes["degree"])




"""sns.countplot(votes["degree"])
plt.show()"""


X=votes.drop("degree" , axis=1)
y=votes["degree"]

sc=StandardScaler()
X=sc.fit_transform(X)

LogRes=LogisticRegression(penalty='none' , solver="lbfgs")
LogRes.fit(X,y)


prop_scores = np.array(LogRes.predict_proba(X)[:,1])
np.savetxt("prop.csv", prop_scores, delimiter=",")

t=np.array(votes["degree"])

coef = LogRes.coef_[0]
print (coef)

#histogram
treat_plt = plt.hist(prop_scores[t==1], fc=(0, 0, 1, 0.5),bins=20,label='Treated')
cont_plt = plt.hist(prop_scores[t==0],fc=(1, 0, 0, 0.5),bins=20,label='Control')
plt.legend()
plt.xlabel('propensity score')
plt.ylabel('number of counties')
plt.show()

#calibration
prob_true, prob_pred = calibration_curve(t, prop_scores , n_bins=15)
epochs=[ind for ind in range(15)]
"""plt.plot(epochs, prob_true, '-b', label="predicted_prob" )
plt.plot(epochs, prob_pred, '-r' , label="real_prob")
plt.legend(loc="lower right")
plt.show()"""

brier=np.sum((np.array(prob_true)-np.array(prob_pred))**2)/15
print(brier)

#choose prop scores only between 0.06 and 0.885
