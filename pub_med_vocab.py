import gensim

model = gensim.models.Word2Vec.load('pubmed_2_1.model')

# vocab_size = len(model.wv.vocab)

# print(vocab_size)

# print(model.wv.vocab[10])

print(model.most_similar(positive=['the'], topn=5))