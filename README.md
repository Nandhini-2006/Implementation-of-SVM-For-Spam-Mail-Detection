# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the Program.

2.Import the necessary packages.

3.Read the given csv file and display the few contents of the data.

4.Assign the features for x and y respectively.

5.Split the x and y sets into train and test sets.

6.Convert the Alphabetical data to numeric using CountVectorizer.

7.Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

8.Find the accuracy of the model.

9.End the Program.

## Program:

Program to implement the SVM For Spam Mail Detection.

Developed by: NANDHINI N

Register Number: 212224040212
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import pandas as pd

data=pd.read_csv("spam.csv",encoding="Windows-1252")

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer()

x_train=cv.fit_transform(x_train)

x_test=cv.transform(x_test)

from sklearn.svm import SVC

svc=SVC()

svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)

y_pred

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,y_pred)

accuracy

## Output:

![Screenshot 2025-05-19 163552](https://github.com/user-attachments/assets/2a6d640b-aca6-4aa4-8ab1-5c00214a31b6)

![Screenshot 2025-05-19 163558](https://github.com/user-attachments/assets/c492f63d-1503-46cd-97d9-98d9841a79e1)

![Screenshot 2025-05-19 163604](https://github.com/user-attachments/assets/768a427b-ce53-4cac-88e9-995586cb34db)

![Screenshot 2025-05-19 163609](https://github.com/user-attachments/assets/4c25de45-cc11-4b5b-9b26-c02507a96bf4)

![Screenshot 2025-05-19 163614](https://github.com/user-attachments/assets/38d737ac-54d8-4c27-88bb-2b96357f0ee2)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
