import math
import numpy as np
import random
import pandas as pd
import copy
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:-1]
del X['Geography']
y = dataset.iloc[:,-1].values

X = X.values

encoder = LabelEncoder()
X[:,1] = encoder.fit_transform(X[:,1])

dataset['Geography'] = encoder.fit_transform(dataset['Geography'])

onehot = OneHotEncoder(handle_unknown='ignore')
X_Onehot = onehot.fit_transform(dataset['Geography'].values.reshape(-1,1)).toarray()
X = np.c_[X,X_Onehot]

minMax = MinMaxScaler()
X = minMax.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

def predict(row,coef):
    x = coef[0]
    for i in range(len(row) - 1):
        x += coef[i+1] * row[i]
    return 1 / (1 + math.exp(-x))

def accuracy_metrics(actual,prediction):
    count = 0
    for i in range(len(actual)):
        if actual[i] == prediction[i]:
            count += 1
    return count/len(actual) * 100

def logistic():
    file = open('weights.pkl','rb')
    coef = pkl.load(file)
    file.close()
    predictions = []
    for row in x_test:
        pred = predict(row,coef)
        predictions.append(round(pred))
    accuracy = accuracy_metrics(y_test,predictions)
    return accuracy

acc = logistic()
print(acc)
