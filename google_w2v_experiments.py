import gensim
import pickle
import operator
from collections import OrderedDict
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
import copy

# model = gensim.models.KeyedVectors.load_word2vec_format('/home/bworkman/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
# model.save('google_w2v.model')
model = gensim.models.Word2Vec.load('google_w2v.model')

sent_1 = "i walked my dog barnie across the busy highway"
sent_2 = "we ran our pooch to the other side of the road"
sent_3 = "i don't like green eggs and ham"

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def sentence_to_vectors(sentence):
    sent = remove_stopwords(sentence)
    words = filter(lambda word: word in model.wv.vocab, sent.split())
    return [(word, model.wv.get_vector(word)) for word in words]

print("==================================")
print(sentence_to_vectors(sent_1))
print("==================================")
print(sentence_to_vectors(sent_2))

def closest_words(sentence_vectors1, sentence_vectors2):

    sentence_vectors2_stack = copy.deepcopy(sentence_vectors2)
    closest_words = []

    for word_vector1 in sentence_vectors1:
        (word1, vector1) = word_vector1

        def find_min(word_vec2):
            (word2, vector2) = word_vec2
            return cosine_similarity(vector1, vector2)
            
        min_word_vec2 = min(sentence_vectors2_stack, key=find_min)

        sentence_vectors2_stack.remove(min_word_vec2)

        closest_words.append((word1, word2, ))

