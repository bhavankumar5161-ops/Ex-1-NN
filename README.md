

<H3>EX. NO.1</H3>

<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:



```
Name : P.Bhavankumar
Reg no. : 212225240026

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import io

df=pd.read_csv('Churn_Modelling.csv')
df

x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)

print(df.isnull().sum())

df.duplicated().sum()

df.drop(['Surname'],axis=1,inplace=True) 
df.drop(['CustomerId','Gender','Geography'],axis=1,inplace=True)
df

df.describe()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

x1=df1.iloc[:,:-1].values
print(x1)
y1=df1.iloc[:,-1].values
print(y1)

x_train,x_test,y_train,y_test=train_test_split(x1,y1,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```


## OUTPUT:


# The dataset:


<img width="1479" height="464" alt="359101053-20d15b52-7c53-4f49-851e-d2cac3981c82" src="https://github.com/user-attachments/assets/7101d14b-ddd0-4c32-b07f-299cba525d74" />


# Splitting the dataset: 


<img width="509" height="194" alt="359101298-d7541306-08ca-429c-a03e-2a7fbf824cd8" src="https://github.com/user-attachments/assets/663ea1e9-1e0d-4647-908d-e544d21ed137" />


# Checking for null values:

<img width="358" height="363" alt="359101437-892f1695-70e7-4ecb-8f8c-a9ec96ac2a87" src="https://github.com/user-attachments/assets/9d3c8a14-f818-4a66-be17-861c72ed57ac" />

# Checking for duplication:

<img width="268" height="72" alt="359101576-2075962b-c497-48d8-b405-f072a1e9c86a" src="https://github.com/user-attachments/assets/d5433165-7a24-4375-afbc-cb8b95fec233" />


# Dropping unwanted features:

<img width="1278" height="575" alt="359101910-12abc2d9-a0b0-4897-9126-8bdb3dc0432b" src="https://github.com/user-attachments/assets/a68c53cc-5ade-4157-9118-80466a86b01e" />


# Describing the dataset:

<img width="1160" height="506" alt="359102057-c79e4e31-71b6-493a-853b-f59b59414d57" src="https://github.com/user-attachments/assets/f8fdf361-6c3e-44c6-81d5-6cc4605a9830" />


# Scaling the values:

<img width="824" height="548" alt="359102167-eb9a7660-0d6d-4698-864e-bf90d144e0f6" src="https://github.com/user-attachments/assets/6ae18dde-60a6-4076-b96b-f96137d2b2ea" />


# X and Y features:

<img width="827" height="451" alt="359102525-316c5ced-4649-4861-babb-bdaeebb47543" src="https://github.com/user-attachments/assets/75675a6f-ada5-4943-bfa8-d2ebbdd69ff9" />


# Splitting the training and testing dataset:

<img width="976" height="481" alt="359102672-5349ba7e-0157-4e43-a195-d472146f9e24" src="https://github.com/user-attachments/assets/dac2dd00-ade3-45b8-a828-2b2305635071" />




## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


