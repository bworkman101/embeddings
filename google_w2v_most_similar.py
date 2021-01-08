import gensim
import pickle
import operator
from collections import OrderedDict

# vocab = pickle.load(open( "vocab.p", "rb" ))
vocab = pickle.load(open( "existing_vocab.p", "rb" ))
print vocab.keys()

model = gensim.models.KeyedVectors.load_word2vec_format('/home/bworkman/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

print "saint st distance =>", model.wv.distance("saint", "st")
print "saint dog distance =>", model.wv.distance("saint", "dog")
print "saint cat distance =>", model.wv.distance("saint", "cat")

# print "baby", model.wv['baby']
# print "newborn", model.wv['newborn']
# print "adult", model.wv['adult']
# print "nino", model.wv['nino']
# print "benjamin", model.wv['benjamin']
# print "coltan", model.wv['coltan']

# given a word
# which word in vocab is most similar

def print_top_cats(search_word):
    similarities = dict([(target, model.wv.similarity(search_word, target)) for target in vocab.keys()])
    similarities_sorted = sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)
    top_20 = dict(similarities_sorted[0:10])

    print top_20

    categories = pickle.load(open("categories.p", "rb"))

    top_20_cat = []
    for category in categories:
        for top, similarity in top_20.iteritems():
            if top in category.split(' '):
                top_20_cat.append((category, similarity))

    top_20_cat = dict(list(OrderedDict.fromkeys(top_20_cat)))

    top_20_cat = sorted(top_20_cat.items(), key=operator.itemgetter(1), reverse=True)
    print "top 20 categories"
    for top_cat in top_20_cat:
        print " -> ", top_cat

exit_me = False

while(exit_me == False):
    search_word = raw_input('search word: ')
    if search_word == 'exit':
        exit_me = True
    else:
        print_top_cats(search_word)
