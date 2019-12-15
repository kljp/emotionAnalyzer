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
txt_label = open('category.txt', 'r')
labels = txt_label.read()
labels = labels.split('\n')
y = labels[:-1]
txt_label.close()
for i in range(0, len(y)):
    if y[i] == 'culture':
        y[i] = '1'
    elif y[i] == 'digital':
        y[i] = '2'
    elif y[i] == 'economic':
        y[i] = '3'
    elif y[i] == 'foreign':
        y[i] = '4'
    elif y[i] == 'nation':
        y[i] = '5'
    elif y[i] == 'politics':
        y[i] = '6'
    elif y[i] == 'society':
        y[i] = '7'
    elif y[i] == 'entertainment':
        y[i] = '8'
    else:
        y[i] = '9'

vocab = []
txt_input = open('output_test.txt', 'r')
docs = txt_input.readlines()
txt_input.close()

for doc in docs:
    line = doc.rstrip('\n')
    line = line.split(' ')
    vocab.append(line[:-1])

scores = [[0 for i in range(100)] for i in range(0, len(vocab))]

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
            y[i] = '10'
            del_index += 1
            break

print(del_index)
if del_index > 0:
    while True:
        try:
            key = y.index('10')
            del scores[key]
            del y[key]
        except:
            break


print(len(scores))
print(len(y))


for i, item in enumerate(scores):
    item.append(y[i])

random.shuffle(scores)

with open('dataset_test.data', 'w', newline='\n') as f:
    writer = csv.writer(f, delimiter=',')
    for data in scores:
        writer.writerow(data)





