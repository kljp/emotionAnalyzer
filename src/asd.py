from konlpy.tag import Twitter, Kkma
from gensim.models import word2vec

model = word2vec.Word2Vec.load("wiki.model")
word = '감사'
twit = Twitter().pos(word)
kkma = Kkma().pos(word)
# print(twit[0][1])
# print(kkma)
print(model.most_similar(positive=[word]))

print(kkma)