# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and preprocess the employee salary dataset (handle missing values, encode categorical data).
2.Split the dataset into features (X) and target (y), then into training and testing sets.
3.Initialize the DecisionTreeRegressor model with appropriate hyperparameters.
4.Train the model using the training dataset. 5.Evaluate the model on the test dataset and predict employee salaries. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: chanthru v
RegisterNumber: 24900997 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
data=pd.read_csv(r"C:\ml experinment\Salary.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
data["left"].value_counts()
 
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
#  print(data.head())
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
#  print(x.head())    
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
#  print(accuracy)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
plot_tree(dt,feature_names=x.columns,class_names=['salary','left'],filled=True)
plt.show()
```
## Output:
![exp 9(1)](https://github.com/user-attachments/assets/02758ca7-b398-4ffc-98e1-f31d80884794)
![exp 9(2)](https://github.com/user-attachments/assets/4c27b78a-3395-40d3-9ec7-b2e9c4de9f8d)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
