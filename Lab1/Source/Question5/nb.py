from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split


# load glass data set
heart = pd.read_csv('heart2.csv')
x = heart[['sex','slope','ca','thal']]
y = heart['target']

# Use cross validation to create training and testing part
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

clf = GaussianNB()
clf.fit(X_train, y_train)
print('Accuracy of Naive Bayes GaussianNB on training set: {:.2f}'.format(clf.score(X_train, y_train)))
# Evaluate the model on testing part
print('Accuracy of Naive Bayes GaussianNB on test set: {:.2f}'.format(clf.score(X_test, y_test)))