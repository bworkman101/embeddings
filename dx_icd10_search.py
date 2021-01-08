import gensim
import numpy as np

model = gensim.models.Word2Vec.load('pubmed_2.model')

# similar = model.most_similar(positive=["shigellosis"], topn=10)

# print(model.wv["shigellosis"])

# .vocab[word].index

# print(np.array2string(next(iter(model.wv.vectors)), separator=',', max_line_width=1000))

# print(next(iter(model.wv.vocab)))

# word_vector = model.wv["shigellosis"]
# word_vector_str = np.array2string(word_vector, separator=',', max_line_width=1000)
# str_left_replace = word_vector_str.replace("[", "")
# str_right_replace = str_left_replace.replace("]", "")
# print("'shigellosis', ", str_right_replace)

with open("word_2_vec_vectors.csv", "w") as outfile:

    for word in model.wv.vocab:
    
        outfile.write("'{}', ".format(word))

        word_vector = model.wv[word]
        word_vector_str = np.array2string(word_vector, separator=',', max_line_width=50000)
        str_left_replace = word_vector_str.replace("[", "")
        str_right_replace = str_left_replace.replace("]", "")

        outfile.write(str_right_replace)

        outfile.write("\n")
