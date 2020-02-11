import math
import numpy as np
import random
import pandas as pd
import copy
import pickle as pkl
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder

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

def cross_validation(k=5):
    X_copy = list(X)
    fold_size = len(X) // k
    folds = []
    for i in range(k):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(0,len(X_copy))
            fold.append(X_copy.pop(index))
        folds.append(fold)
    return folds

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

def gradientDescent(x_train,y_train,epochs,alpha):
    coef = np.zeros(x_train.shape[1] + 1)
    n = len(x_train)
    for i in range(epochs):
        predictions = []
        for j in range(len(x_train)):
            pred = predict(x_train[j],coef)
            error = pred - y_train[j]
            coef[0] = coef[0] - alpha * (1/n) * np.sum(error)
            for k in range(len(x_train[0] - 1)):
                coef[k+1] = coef[k+1] - alpha * (1/n) * np.dot(x_train[j][k],error)

        print("{} Epochs Completed".format(i))
        '''
        if i % 10 == 0:
            for row in x_train:
                pred = predict(row,coef)
                predictions.append(round(pred))
            accuracy = accuracy_metrics(y_train,predictions)
            print("Accuracy is",accuracy)
        '''
    return coef

def evaluateAlgorithm(epochs,alpha):
    folds = cross_validation()
    scores = []
    for fold in folds:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        train = list(folds)
        train.remove(fold)
        for train_fold in train:
            for data in train_fold:
                x_train.append(data[:-1])
                y_train.append(data[-1])
        
        for data in fold:
            x_test.append(data[:-1])
            y_test.append(data[-1])
        x_train = np.asarray(x_train)
        x_test = np.asarray(x_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)
        score = logistic(x_train,y_train,x_test,y_test,epochs,alpha)
        print("Score is",score)
        scores.append(score)
        return scores
    
def logistic(x_train,y_train,x_test,y_test,epochs,alpha):
    coef = gradientDescent(x_train,y_train,epochs,alpha)
    file = open('weights.pkl','wb')
    pkl.dump(coef,file)
    file.close()
    predictions = []
    for row in x_test:
        pred = predict(row,coef)
        predictions.append(round(pred))
    accuracy = accuracy_metrics(y_test,predictions)
    return accuracy

epochs = 50
alpha = 0.01

scores = evaluateAlgorithm(epochs,alpha)
print(scores)
