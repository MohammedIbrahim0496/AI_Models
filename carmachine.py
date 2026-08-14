import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
cd=pd.read_csv("car_prediction_data.csv")
cd.replace({"Fuel_Type":{"Petrol":0,"Diesel":1,"CNG":2}},inplace=True)
cd.replace({"Transmission":{"Manual":0,"Automatic":1}},inplace=True)
cd.replace({"Seller_Type":{"Dealer":0,"Individual":1}},inplace=True)
x=cd.drop(["Car_Name","Selling_Price"],axis=1)
y=cd["Selling_Price"]
xer=[2017,0.51,4300,0,0,0,0]
xer=np.asarray(xer).reshape(1,-1)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=2)
len=LinearRegression()
model=len.fit(xtrain,ytrain)
a=model.predict(xer)
print(a)
#joblib.dump(model,"Carmodel.pkl")