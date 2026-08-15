import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
import joblib
dd=pd.read_csv(r"C:\Users\nmoha\OneDrive\Desktop\diabetes\diabetes.csv")
x =dd.drop(columns="Outcome",axis=1)
y =dd["Outcome"]
s=StandardScaler()
x=s.fit_transform(x)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)
model =svm.SVC(kernel="linear")
model.fit(xtrain,ytrain)
'''xt=model.predict(xtest)
yt=accuracy_score(xt,ytest)
print(yt)'''
joblib.dump(s,"diabetes_scaler")
joblib.dump(model,"diabetes_model")
'''c=(10,168,74,0,0,38,0.537,34)
c=np.asarray(c)
c=c.reshape(1,-1)
c=s.transform(c)
b=model.predict(c)
print(b)'''