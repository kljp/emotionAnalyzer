import pickle
import numpy as np
from konlpy.tag import Twitter
from gensim.models import word2vec
from sklearn.preprocessing import StandardScaler

model = word2vec.Word2Vec.load("wiki.model")
ml = pickle.load(open('model.pkl', 'rb'))

label = {1: '기쁨', 2: '슬픔', 3: '놀람', 4: '공포', 5: '혐오', 6: '분노', 7: '알 수 없음'}

sentence = []
input_sentence = '고가 철거하는것 빼곤 한것이 없는데...3선을 바라보시나이까.. 박시장님.   박수칠때 물러나는 아름다운것이 아닐련지'
twit = Twitter().pos(input_sentence)
print(twit)
for i in range(0, len(twit)):
    if twit[i][1] == 'Noun' or twit[i][1] == 'Adjective' or twit[i][1] == 'Verb':
        sentence.append(twit[i][0])
score = [0 for i in range(0, 100)]
count = 0
for word in sentence:
    try:
        sList = model[word]
        print(word, ' ', end='')
        for i in range(0, 100):
            score[i] += sList[i]
        count += 1
    except:
        pass
print()
for i in range(0, 100):
    try:
        score[i] = score[i] / count
    except:
        print('Invalid sentence')
        exit(0)
X = [score]
# sc = StandardScaler()
# X_test_std = sc.transform(X)
y_pred = ml.predict(X)
print(label[int(y_pred[0])])
