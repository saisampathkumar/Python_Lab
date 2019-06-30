import pandas as pd
import numpy as np

train = pd.read_csv('diabetes.csv')

##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().dropna()

#Before EDA
y = data.iloc[:].values
X = data.drop(['outcome'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

#Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])

corr = numeric_features.corr()
print (corr['outcome'].sort_values(ascending=False)[:3], '\n')
'''print (corr['quality'].sort_values(ascending=False)[-3:])'''

##Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

#Remove null values with mean
modifiedDataset = train.fillna(train.mean())

nulls1 = pd.DataFrame(modifiedDataset.isnull().sum().sort_values(ascending=False)[:25])
nulls1.columns = ['Null Count']
nulls1.index.name = 'Feature'
print(nulls1)

data1 = modifiedDataset.select_dtypes(include=[np.number]).interpolate().dropna()

#After EDA
##Build a linear model
y = data1.iloc[:].values
X = data1.drop(['outcome'], axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))