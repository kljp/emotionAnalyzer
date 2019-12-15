import pickle
import numpy as np
from konlpy.tag import Twitter
from gensim.models import word2vec
from sklearn.preprocessing import StandardScaler

model = word2vec.Word2Vec.load("wiki.model")
ml = pickle.load(open('model.pkl', 'rb'))

category = {1: 'culture', 2: 'digital', 3: 'economic', 4: 'foreign', 5: 'nation', 6: 'politics', 7: 'society',
         8: 'entertainment'}
label = {1: '기쁨', 2: '슬픔', 3: '놀람', 4: '공포', 5: '혐오', 6: '분노', 7: '알 수 없음'}

data = []
lines = open('dataset_test.data', 'r').readlines()
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
for i in range(0, len(X)):
    for j in range(0, 100):
        X[i][j] = float(X[i][j])

y_pred = ml.predict(X)
cat_scores = [[0 for i in range(0, 5)] for i in range(0, 8)]
for i, p in enumerate(y_pred):
    a = label[int(y_pred[i])]
    if a == '기쁨':
        cat_scores[int(y[i]) - 1][0] += 1
    elif a == '슬픔':
        cat_scores[int(y[i]) - 1][1] += 1
    elif a == '혐오':
        cat_scores[int(y[i]) - 1][2] += 1
    elif a == '분노':
        cat_scores[int(y[i]) - 1][3] += 1
    cat_scores[int(y[i]) - 1][4] += 1

total = 0
for i in range(0, 8):
    print('[' + category[i + 1] + ']')
    try:
        print('기쁨:', str(cat_scores[i][0]) + '개 / ' + str(cat_scores[i][0] / cat_scores[i][4]) + '%')
    except ZeroDivisionError:
        print('기쁨: 0%')
    try:
        print('슬픔:', str(cat_scores[i][1]) + '개 / ' + str(cat_scores[i][1] / cat_scores[i][4]) + '%')
    except ZeroDivisionError:
        print('슬픔: 0%')
    try:
        print('혐오:', str(cat_scores[i][2]) + '개 / ' + str(cat_scores[i][2] / cat_scores[i][4]) + '%')
    except ZeroDivisionError:
        print('혐오: 0%')
    try:
        print('분노:', str(cat_scores[i][3]) + '개 / ' + str(cat_scores[i][3] / cat_scores[i][4]) + '%')
    except ZeroDivisionError:
        print('분노: 0%')
    print('총', str(cat_scores[i][4]) + '개')
    total += cat_scores[i][4]
print('\n총 댓글 수:', str(total))








