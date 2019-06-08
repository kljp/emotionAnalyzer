from gensim.models import word2vec

model = word2vec.Word2Vec.load("wiki.model")
# print(model.most_similar(positive=[""]))
try:
    print(model['모르시오라'])
except:
    print('None')