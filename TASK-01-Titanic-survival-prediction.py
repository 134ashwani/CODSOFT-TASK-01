import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv('Titanic-Dataset.csv')
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
titanic['Sex']=labelencoder.fit_transform(titanic['Sex'])
titanic=titanic.drop(['Age'],axis= 1)
titanic_final = titanic
print(titanic_final.head(10))
# Model Training
a=titanic[['Pclass','Sex']]
b=titanic['Survived']
from sklearn.model_selection import train_test_split
a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
log =LogisticRegression(random_state=0)
log.fit(a_train,b_train)


# Predicting The Model

pred = print(log.predict(a_test))
print(b_test)

import warnings
warnings.filterwarnings("ignore")
res = log.predict([[int(input("enter the passenger class:")),int(input("enter the sex of the passenger:"))]])
if(res==0):
    print("Sorry! not survived")
else:
    print("survived")