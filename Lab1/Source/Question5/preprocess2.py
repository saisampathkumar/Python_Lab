import pandas as pd
##reading data set
train_df = pd.read_csv('heart.csv')

##Analyze by pivoting features ---> the higher the number means more correlation with the target
print(train_df[['age', 'target']].groupby(['age'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['sex', 'target']].groupby(['sex'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['cp', 'target']].groupby(['cp'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['trestbps', 'target']].groupby(['trestbps'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['chol', 'target']].groupby(['chol'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['fbs', 'target']].groupby(['fbs'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['restecg', 'target']].groupby(['restecg'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['thalach', 'target']].groupby(['thalach'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['exang', 'target']].groupby(['exang'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['oldpeak', 'target']].groupby(['oldpeak'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['slope', 'target']].groupby(['slope'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['ca', 'target']].groupby(['ca'], as_index=False).mean().sort_values(by='target', ascending=False))
print(train_df[['thal', 'target']].groupby(['thal'], as_index=False).mean().sort_values(by='target', ascending=False))

# ##Correcting by dropping features
train_df = train_df.drop(['age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak'], axis=1)

print(train_df.shape)
train_df.to_csv('heart2.csv',index=False)