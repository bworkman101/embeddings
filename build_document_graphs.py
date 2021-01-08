import gensim
import numpy as np
from scipy import sparse
from bisect import bisect_left
from scipy.sparse import save_npz
from scipy import linalg
import os

"""
Create document graphs.

? would puncuation matter

Reading the text from left to right.
Record each word as a node.
Each edge is the connection from that word to the next word.

Example:
   sentence:  I ran to the park, then to the river.
  word_vecs:  0 1   2  3   4     5    2  3   6

  graph:
                      | -> 6 
                      |  
       0 -> 1 -> 2 -> 3 -> 4 -> 5
                 |<-------------|

Graph naming conventions are based on conventions used in GCN
https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
https://towardsdatascience.com/using-graph-convolutional-neural-networks-on-structured-documents-for-information-extraction-c1088dcd2b8f
                                 
"""

model = gensim.models.Word2Vec.load('pubmed_2_1.model')

corpus_files = ['biorxiv_medrxiv', 
                'comm_use_subset', 
                'noncomm_use_subset', 
                'pmc_custom_license']

base = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/texts_clean/'
graph_base = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/graphs'

def index(a, x):
    """
    Search for the index of x in list a.   a must be a sorted list.
    Returns -1 if not found, index of x if found.
    """
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1

def get_word_vec(word):
    if word in model.wv.vocab:
        return model.wv.word_vec(word)
    else:
        # not sure what having a word vector of zeros will have
        return np.zeros(model.wv.vector_size)

read_first = False

def relu(X):
   return np.maximum(0,X)

def corpus_to_graph(corpus_doc):
    """
    This uses the mean rule for aggregation.
    See https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-62acf5b143d0
    """
    words = corpus_doc.split(' ')
    num_words = len(words)

    # get distinct sorted words
    words_ds = list(set(words))
    words_ds.sort()
    n = len(words_ds)

    # A_hat - n*n matrix to describe connections
    #   we skip creating A and I and jump straight to A_hat
    #   A_hat is adding A node connection matrix to I identity matrix
    A_hat = sparse.eye(n).todok()

    for pos in range(num_words):
        word = words[pos]
        word_idx = index(words_ds, word)

        if pos + 1 < num_words:
            n_word = words[pos + 1]
            n_word_idx = index(words_ds, n_word)
            A_hat[word_idx, n_word_idx] = 1

    # D - degree of nodes matrix
    D = A_hat.sum(0)[0] # sum will make this a numpy matrix
    D = np.squeeze(np.asarray(D)) # make it a 1d array
    D = np.matrix(np.diag(D))

    # X - create a feature matrix, should just be vectors for each node
    X = []

    for pos in range(n):
        word = words_ds[pos]
        word_vector = get_word_vec(word)
        X.append(word_vector)

    X = np.array(X)

    layer_1_size = 10
    layer_2_size = 3
    layer_3_size = 1

    W_1 = np.random.normal(loc=0, scale=1, size=(X.shape[1], layer_1_size))
    W_2 = np.random.normal(loc=0, size=(W_1.shape[1], layer_2_size))
    W_3 = np.random.normal(loc=0, size=(W_2.shape[1], layer_3_size))

    # print("mat_D.shape", mat_D.shape)
    # print("mat_A.shape", mat_A.shape)
    # print("mat_X.shape", mat_X.shape)
    # print("shape(np.linalg.inv(mat_D) * mat_A * mat_X)", (np.linalg.inv(mat_D) * mat_A * mat_X).shape)
    # print("W_1.shape", W_1.shape)

    # scipy.linalg.fractional_matrix_power(D, -0.5)

    # Mean Rule
    # D_hat = linalg.fractional_matrix_power(D, -1.)
    # l1_out = relu(D_hat * A_hat * X * W_1)
    # l2_out = relu(D_hat * A_hat * l1_out * W_2)
    # l3_out = relu(D_hat * A_hat * l2_out * W_3)

    # Spectral Rule
    D_hat = linalg.fractional_matrix_power(D, -0.5)
    l1_out = relu(D_hat * A_hat * D_hat * X * W_1)
    l2_out = relu(D_hat * A_hat * D_hat * l1_out * W_2)
    l3_out = relu(D_hat * A_hat * D_hat * l2_out * W_3)

    return (A_hat, X, n, D, W_1, W_2, l1_out, l2_out, l3_out)

# for corpus_file in corpus_files:
#     print("converting {} to graph".format(corpus_file))
#     try:
#         graph_dir = "{}/{}".format(graph_base, corpus_file)
#         os.mkdir(graph_dir)
#     except FileExistsError:
#         print("     directory " , graph_dir ,  " already exists")

#     with open("{}{}.text".format(base, corpus_file), 'r') as corpus_docs:
#         file_itr = 0
                
#         for corpus_doc in corpus_docs:
#             dir_tree = int(file_itr / 100)

#             if file_itr % 100 == 0:
#                 try:
#                     graph_dir = "{}/{}/{}".format(graph_base, corpus_file, dir_tree)
#                     os.mkdir(graph_dir)
#                 except FileExistsError:
#                     print("     directory " , graph_dir ,  " already exists")

#             (A, X, n, D) = corpus_to_graph(corpus_doc)

#             a_mat_file = "{}/{}/{}/A_{}".format(graph_base, corpus_file, dir_tree, file_itr)
#             save_npz(a_mat_file, A.tocoo())

#             x_mat_file = "{}/{}/{}/X_{}".format(graph_base, corpus_file, dir_tree, file_itr)
#             np.save(x_mat_file, X)

#             d_mat_file = "{}/{}/{}/D_{}".format(graph_base, corpus_file, dir_tree, file_itr)
#             np.save(d_mat_file, D)

#             file_itr = file_itr + 1

#             if file_itr % 100 == 0:
#                 print("  finished {}".format(file_itr))

#             if read_first == True:
#                 break
#         if read_first == True:
#             break

doc = "hello to the world and all the people, all the friendly people"
(A, X, n, D, W_1, W_2, l1, l2, l3) = corpus_to_graph(doc)

print("[A] ---------------------------------------")
print(" |0|1|2|3|4|5|6")
print("---------------")
for i in range(n):
    print(i, end='|')
    for j in range(n):
        print(A[i, j], end=',')
    print()
print("[X] ---------------------------------------")
print(X)

# print("[W_1]---------------------------------------")
# print(W_1)
# print(W_1.shape)

# print("[W_2]---------------------------------------")
# print(W_2)
# print(W_2.shape)

# print("[l1]---------------------------------------")
# print(l1)
# print(l1.shape)

print("[l2]---------------------------------------")
print(l2)
print(l2.shape)

# print("[l3]---------------------------------------")
# print(l3)
# print(l3.shape)
