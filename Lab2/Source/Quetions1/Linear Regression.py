import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD

plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
tbCallBack= keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,write_graph=True, write_images=True)

train = pd.read_csv('RentData.csv')

#Working with Numeric Features
numeric_features = train.select_dtypes(include=[np.number])
corr = numeric_features.corr()
print (corr['price_aprox_usd'].sort_values(ascending=False)[:3], '\n')

##Null values
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)

##handling missing value
data = train.select_dtypes(include=[np.number]).interpolate().fillna(0)
print(sum(data.isnull().sum() != 0))
print(data)
nulls = pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
print(nulls)
data.to_csv('Modified_Rent_data.csv',index=False)

##Build a linear model
y = data.drop(['price_aprox_local_currency','price','surface_total_in_m2','surface_covered_in_m2','price_usd_per_m2','floor','expenses'],axis=1)

X = data.drop(['price_aprox_local_currency','price','price_aprox_usd','floor','expenses'], axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=70, test_size=.30)
from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)


##Evaluate the performance and visualize results
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))

actual_values = y_test
plt.scatter(predictions,actual_values, alpha=.75,
            color='b')  # alpha helps to show overlapping data
plt.xlabel('Predicted price_aprox_usd')
plt.ylabel('Actual price_aprox_usd')
plt.title('Linear Regression Model')
plt.show()