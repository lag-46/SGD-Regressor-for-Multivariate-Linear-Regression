# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm


1.Load California Housing dataset and convert it into DataFrame.

2.Select independent variables X and dependent variables Y (AveOccup, HousingPrice).

3.Split dataset into training and testing sets.

4.Standardize X and Y values using Standard Scaler using formula:


  Z = (X − Mean) / Standard Deviation

5.Initialize SGD Regression model parameters.

6.Predict output using linear model formula:
               
  Ypred = XW + b

7.Calculate error using:
    
  Error = Ypred − Y

8.Update weights using SGD update rule:
   
   
   W = W − learning_rate * (Error * X)

9.Repeat prediction and weight update until maximum iterations reached.

10.Predict outputs for test data using trained model:


  Ypred = Xtest * W + b

11.Convert predicted scaled values back to original values using inverse scaling:


  X = (Z * Standard Deviation) + Mean

12.Calculate Mean Squared Error (MSE) using formula:


  MSE = (1/n) * Σ (Yactual − Ypred)²

13.Display predicted output values.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: PANDEESWARAN N
RegisterNumber:  212224230191
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

df.info()

X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y=df[['AveOccup','HousingPrice']]
Y.info()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print(Y_pred)

```

## Output:

## Head :

<img width="817" height="313" alt="image" src="https://github.com/user-attachments/assets/9d7257fc-722b-42b0-be28-2f7d9cf14d16" />

## Dataset info : 

<img width="547" height="368" alt="image" src="https://github.com/user-attachments/assets/9961ff16-ce0c-4efe-b901-4ca34d1e3810" />

## Removing columns :

<img width="532" height="329" alt="image" src="https://github.com/user-attachments/assets/672e008b-b92e-41e6-937c-658e39c584a0" />

## Columns info :

<img width="551" height="224" alt="image" src="https://github.com/user-attachments/assets/deb51d8d-d91f-455c-b22a-2b469721abc3" />

## X_train,X_test,Y_train,Y_test :

<img width="693" height="791" alt="image" src="https://github.com/user-attachments/assets/7b22ac01-5afe-4968-a907-423034c1ca50" />

## MultiOutputRegressor(sgd) :

<img width="317" height="195" alt="image" src="https://github.com/user-attachments/assets/7eca7c61-4928-4043-b56b-8c4673f51fb0" />

## Mean Squared error :

<img width="381" height="172" alt="image" src="https://github.com/user-attachments/assets/18c093c0-784f-4ccc-ad59-198fe5454f34" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
