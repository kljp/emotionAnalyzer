from gensim.models import word2vec
import csv
import random

model = word2vec.Word2Vec.load("wiki.model")

# y = []
# txt_label = open('label.txt', 'r')
# labels = txt_label.read()
# labels = labels.split('\n')
# y = labels[:-1]
# txt_label.close()

y = []
txt_label = open('label.txt', 'r')
labels = txt_label.read()
labels = labels.split('\n')
y = labels[:-1]
txt_label.close()
for i in range(0, len(y)):
    if y[i] == '기쁨':
        y[i] = '1'
    elif y[i] == '슬픔':
        y[i] = '2'
    elif y[i] == '놀람':
        y[i] = '3'
    elif y[i] == '공포':
        y[i] = '4'
    elif y[i] == '혐오':
        y[i] = '5'
    elif y[i] == '분노':
        y[i] = '6'
    else:
        y[i] = '7'

vocab = []
txt_input = open('output.txt', 'r')
docs = txt_input.readlines()
txt_input.close()

for doc in docs:
    line = doc.rstrip('\n')
    line = line.split(' ')
    vocab.append(line[:-1])

scores = [[0 for i in range(100)] for i in range(0, len(vocab))]

# del_index = []
# for i, sen in enumerate(vocab):
#     count = 0
#     for word in sen:
#         try:
#             sList = model[word]
#             for j in range(100):
#                 scores[i][j] += sList[j]
#             count += 1
#         except:
#             pass
#
#     for j in range(100):
#         try:
#             scores[i][j] = scores[i][j] / count
#         except:
#             del_index.append(i)
#             break
#
# print(len(del_index))
# if len(del_index) > 0:
#     for index in del_index:
#         del scores[index]
#         del y[index]
#         temp_del_index = del_index[0]
#         del del_index[0]
#         for i in range(0, len(del_index)):
#             if del_index[i] > temp_del_index:
#                 del_index[i] -= 1

print(len(scores))
print(len(y))
del_index = 0
for i, sen in enumerate(vocab):
    count = 0
    for word in sen:
        try:
            sList = model[word]
            for j in range(100):
                scores[i][j] += sList[j]
            count += 1
        except:
            pass

    for j in range(100):
        try:
            scores[i][j] = scores[i][j] / count
        except:
            y[i] = '8'
            del_index += 1
            break

print(del_index)
if del_index > 0:
    while True:
        try:
            key = y.index('8')
            del scores[key]
            del y[key]
        except:
            break


print(len(scores))
print(len(y))

# scores2 = scores
# scores3 = scores
# scores4 = scores
# scores5 = scores
# y2 = y
# y3 = y
# y4 = y
# y5 = y
# for i in range(0, len(scores)):
#     for j in range(0, len(scores[i])):
#         scores2[i][j] = scores2[i][j] * 1.1
#         scores3[i][j] = scores3[i][j] * 0.9
#         scores4[i][j] = scores4[i][j] * 1.2
#         scores5[i][j] = scores5[i][j] * 0.8
# scores.extend(scores2)
# scores.extend(scores3)
# scores.extend(scores4)
# scores.extend(scores5)
# y.extend(y2)
# y.extend(y3)
# y.extend(y4)
# y.extend(y5)

for i, item in enumerate(scores):
    item.append(y[i])

random.shuffle(scores)

with open('dataset.data', 'w', newline='\n') as f:
    writer = csv.writer(f, delimiter=',')
    for data in scores:
        writer.writerow(data)

# data = []
# lines = open('dataset.data', 'r').readlines()
# for line in lines:
#     line = line.rstrip('\n')
#     data.append(line.split(','))
#
# X = []
# y = []
# for item in data:
#     X.append(item[:100])
#     y.append(item[-1])





