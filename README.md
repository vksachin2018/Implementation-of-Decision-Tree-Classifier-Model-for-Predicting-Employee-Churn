# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: gokul sachin.k
RegisterNumber: 212223220025
 
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Data Head:
![169693675-2a2f8bd7-9a87-49dc-a58c-777969b5f353](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/6b176c80-138f-46d4-b817-57ab91f33876)
## Information:
![169693680-b6183dca-cdfb-4dad-afef-3badcecd05f9](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/70585683-c693-4c14-abd5-53791d29009e)
## Null dataset:
![169693714-10634ad2-5b16-4db4-8b72-3d7b3babd95f](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/707f5a65-74c5-4053-9b51-2d6af2cf05e5)
## Value_counts():
![169693730-1efadbf5-4cec-4d2b-bbdd-5d29fcaddc36](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/d7642773-1efb-459f-a828-60408ce7f9cd)
## Data Head:
![169693736-5f392e94-f043-40fa-a0ed-32e89ad2ddb0](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/2ce60394-d13c-4191-911b-3539d05c9872)
## x.head():
![169693739-0365b04f-731b-404b-b914-ef3b5b57c3cf](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/4c9446ac-8490-4d84-86f3-5855b95a62c3)
## Accuracy:
![169693745-cd8c6451-7622-4ef9-a65c-3d7e3bd661de](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/87d69401-c96b-4022-95f3-222a9bed1dfc)
## Data Prediction:
![169693750-5106819e-ba64-4653-ad7b-b0f06df09a72](https://github.com/vksachin2018/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/149366019/83f6b910-b5e2-4e2e-b240-ee1dcf65a70e)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using 
python programming.
