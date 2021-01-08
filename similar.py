import gensim

model = gensim.models.Word2Vec.load('vitaminD.model')

for word in model.wv.most_similar(positive='methylcobalamin', topn=50):
    print word