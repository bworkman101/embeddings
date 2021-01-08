import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import math

model = gensim.models.KeyedVectors.load_word2vec_format('/home/bworkman/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

king_vector = model.wv.get_vector("king")
queen_vector = model.wv.get_vector("queen")
man_vector = model.wv.get_vector("man")
woman_vector = model.wv.get_vector("woman")
tv_vector = model.wv.get_vector("tv")
germany_vector = model.wv.get_vector("germany")
cloud_vector = model.wv.get_vector("cloud")

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def cosine_distance(vec1, vec2):
    return 1. - cosine_similarity(vec1, vec2)

print("king_vector", king_vector)
print("queen_vector", queen_vector)
print("man_vector", man_vector)
print("woman_vector", woman_vector)

# Man is to king as woman is to 

# Man + King = Woman + x
# King - Man + Woman = x

def n(vector):
    return vector / np.linalg.norm(vector)

x = n(king_vector) + (n(woman_vector) - n(man_vector))

# x = np.mean(np.array([king_vector, man_vector * -1.0, woman_vector]), axis=0)

result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

print("x", x)
# print(".word_vec(word, use_norm=True))", model.word_vec("king", use_norm=True))
print("result", result)
print("★★ x woman similarity ★★", cosine_similarity(x, n(woman_vector)))
print("x king similarity", cosine_similarity(x, n(king_vector)))
print("x queen similarity", cosine_similarity(x, n(queen_vector)))
print("x man similarity", cosine_similarity(x, n(man_vector)))
print("x tv similarity", cosine_similarity(x, n(tv_vector)))
print("x germany similarity", cosine_similarity(x, n(germany_vector)))
print("x cloud similarity", cosine_similarity(x, n(cloud_vector)))
