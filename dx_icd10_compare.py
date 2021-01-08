import gensim
import numpy as np

model = gensim.models.Word2Vec.load('pubmed.model')

similar = model.most_similar(positive=["broken"], topn=10)

model.evaluate_word_pairs()

print(similar)