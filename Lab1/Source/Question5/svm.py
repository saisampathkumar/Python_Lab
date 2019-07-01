
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

# load glass data set
heart = pd.read_csv('heart2.csv')
x = heart[['sex','slope','ca','thal']]
y = heart['target']

# Use cross validation to create training and testing part
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Implement linear SVM method using scikit library
svm = LinearSVC(random_state=0, tol=1e-5)
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'.format(svm.score(X_train, y_train)))
# test data set acc
print('Accuracy of SVM classifier on test set: {:.2f}'.format(svm.score(X_test, y_test)))
