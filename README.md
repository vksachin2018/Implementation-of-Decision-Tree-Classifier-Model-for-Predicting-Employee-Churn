# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas 

2. Import Decision tree classifier

3. Fit the data in the model
   
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Gokul sachin k
RegisterNumber:  212223220025
*/
```
```python
import pandas as pd
df=pd.read_csv("/content/Employee.csv")
print("data.head():")
df.head()
```
```python
print("data.info()")
df.info()
```
```python
print("data.isnull().sum()")
df.isnull().sum()
```
```python
print("data value counts")
df["left"].value_counts()
```
```python
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
print("data.head() for Salary:")
df["salary"]=le.fit_transform(df["salary"])
df.head()
```
```python
print("x.head():")
x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
```python
y=df["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
```
```python
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
```python
print("Data prediction")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plot_tree(dt,filled=True,feature_names=x.columns,class_names=['salary' , 'left'])
plt.show()
```

## Output:
<img width="1499" height="308" alt="436157065-729f156a-2aa9-4d44-a881-11aa0ee8b4b2" src="https://github.com/user-attachments/assets/f52b3635-0550-4f35-9955-93e59328457c" />

<img width="576" height="403" alt="436157150-e59695a1-040c-4334-a87c-2d7bff338ad3" src="https://github.com/user-attachments/assets/a4ddc184-a605-48f5-8694-67a0dfad1cb5" />
<img width="325" height="502" alt="436157241-54891a24-e47c-4470-9d52-11f394d7ea38" src="https://github.com/user-attachments/assets/bcc15b40-8234-4d64-9612-48f5dba10494" />

<img width="228" height="233" alt="436157314-27216ab2-c2c8-4c75-90b3-b9111435a84f" src="https://github.com/user-attachments/assets/a00530ef-6bae-4ac8-896e-0c9193043e26" />
<img width="1484" height="296" alt="436157422-a1031a8a-6a2f-4edb-afe4-771ba7f501ae" src="https://github.com/user-attachments/assets/bc189ac0-093c-4e96-a665-1324fbf232c8" />

<img width="1433" height="286" alt="436157499-c1e7f6e6-ceaa-446a-8f82-a0294f4c8130" src="https://github.com/user-attachments/assets/aaadc61a-9ee0-4e98-ab20-3208f8fb79a2" />
<img width="358" height="47" alt="436157592-8accd484-eaea-44e2-814a-1708afd982b9" src="https://github.com/user-attachments/assets/b2c1537a-875a-413f-abc4-389e1b0fadc4" />

<img width="244" height="57" alt="436157656-3c77533f-79e3-4580-a772-1e03bafff350" src="https://github.com/user-attachments/assets/484f4594-5a4f-4113-8659-7e336910bd9d" />



<img width="817" height="797" alt="436157943-6c9ac447-f3b2-4049-ac26-f0eab4250f37" src="https://github.com/user-attachments/assets/642107af-ccd0-4ffb-ba01-dc3c8c6deeb9" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
