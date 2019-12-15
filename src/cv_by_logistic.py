from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import KFold, cross_val_score

if __name__ == '__main__':

    data = []
    lines = open('dataset.data', 'r').readlines()
    for line in lines:
        line = line.rstrip('\n')
        data.append(line.split(','))
    # for i in range(0, len(data)):
    #     print(data[i])
    X = []
    y = []
    for item in data:
        X.append(item[:100])
        y.append(item[-1])

    ml = LogisticRegression(C=0.01, random_state=0, solver='saga')
    k_fold = KFold(n_splits=5)
    scores = np.zeros(5)

    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        for index in train_index:
            X_train.append(X[index])
            y_train.append(y[index])
        for index in test_index:
            X_test.append(X[index])
            y_test.append(y[index])
        ml.fit(X_train, y_train)
        y_pred = ml.predict(X_test)
        scores[i] = accuracy_score(y_test, y_pred)
        print(scores[i])

    # cross_val_score(ml, X, y, cv=k_fold)