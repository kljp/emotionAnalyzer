from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np


lines = open('dataset.data', 'r').readlines()
lower = [0, 2529, 5058, 7587]
upper = [2528, 5057, 7586, 10115]
loss = []
accuracy = []

for i in range(4):
    data = []
    data2 = []
    count = 0
    for line in lines:
        if count >= lower[i] and count <= upper[i]:
            line = line.rstrip('\n')
            data2.append(line.split(','))
        else:
            line = line.rstrip('\n')
            data.append(line.split(','))
        count = count + 1

    X = []
    y = []
    X_test = []
    y_test = []
    for item in data:
        X.append(item[:100])
        y.append(str(int(item[-1]) - 1))
    for item in data2:
        X_test.append(item[:100])
        y_test.append(str(int(item[-1]) - 1))

    X = np.array(X)
    y = np.array(y)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y = np_utils.to_categorical(y, 8)
    y_test = np_utils.to_categorical(y_test, 8)

    model = Sequential()

    model.add(Dense(512, input_shape=(100,)))
    model.add(Activation('relu'))
    model.add(Dropout(0, 2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0, 2))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    hist = model.fit(X, y)

    score = model.evaluate(X_test, y_test, verbose=1)
    print('loss = ', score[0])
    print('accuracy = ', score[1])
    loss.append(score[0])
    accuracy.append(score[1])

sum_loss = 0
sum_accuracy = 0
for i in range(4):
    sum_loss = sum_loss + loss[i]
    sum_accuracy = sum_accuracy + accuracy[i]

print("-------------------------------")
print("after 4-fold cross validation")
print('loss = ', sum_loss / 4)
print('accuracy = ', sum_accuracy / 4)