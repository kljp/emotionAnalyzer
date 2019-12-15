from sklearn import datasets
from sklearn.cross_validation import  train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import style
from mylib.plotdregion import plot_decision_region
import pickle

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # ml = LogisticRegression(C=1000, random_state=0)
    ml = LogisticRegression(C=0.01, random_state=0, solver='saga')
    ml.fit(X_train_std, y_train)
    y_pred = ml.predict(X_test_std)

    print('총 테스트 개수: %d, 오류개수:%d' % (len(y_test), (y_test != y_pred).sum()))
    print('정확도: %2f' % accuracy_score(y_test, y_pred))

    pickle.dump(ml, open('model.pkl', 'wb'))

    # X_combined_std = np.vstack((X_train_std, X_test_std))
    # y_combined_std = np.hstack((y_train, y_test))
    # plot_decision_region(X=X_combined_std, y=y_combined_std, classifier=ml,
    #                      test_idx=range(105, 150), title='scikit-learn Logistic Regression')