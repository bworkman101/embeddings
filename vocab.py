import gensim

model = gensim.models.Word2Vec.load('pubmed.vocab_2.model')

# print model.wv.index2word

print(len(model.wv.vocab))