import gensim
from gensim.models.word2vec import FAST_VERSION

print("FAST_VERSION", FAST_VERSION)

# archive_files = [('comm_use.A-B', 248270),
#                  ('comm_use.C-H', 248296),
#                  ('comm_use.I-N', 260121),
#                  ('comm_use.O-Z', 479198),
#                  ('non_comm_use.A-B', 122730),
#                  ('non_comm_use.C-H', 155319),
#                  ('non_comm_use.I-N', 308762),
#                  ('non_comm_use.O-Z', 125295)]  # 1,947,991

archive_files = [('biorxiv_medrxiv', 803),
                 ('comm_use_subset', 9000),
                 ('noncomm_use_subset', 1973),
                 ('pmc_custom_license', 1426)]

# base = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/datasets/pubmed/articles/corpus/'
base = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/cord19/texts_clean/'

# model = gensim.models.Word2Vec(
#     window=35,
#     min_count=2,
#     workers=100,
#     sg=1)

# not_first_run = False

# for corpus_tuple in archive_files:
#     (corpus_name, num_lines) = corpus_tuple
#     corpus_file = '{}{}.txt'.format(base, corpus_name)
#     print("building vocab", corpus_file)
#     model.build_vocab(corpus_file=corpus_file, update=not_first_run, progress_per=1000)
#     not_first_run = True

# model.save('pubmed.vocab_2.model')

# model = gensim.models.Word2Vec.load('pubmed.vocab_2.model')

# for corpus_tuple in archive_files:
#     (corpus_name, num_lines) = corpus_tuple
#     corpus_file = '{}{}.txt'.format(base, corpus_name)
#     print("training", corpus_file)
#     model.train(corpus_file=corpus_file, total_examples=num_lines, epochs=model.epochs, total_words=model.corpus_total_words)

# model.save('pubmed_2.model')

# -------------------------------------------------------------------------------------
# updating   https://stackoverflow.com/questions/22121028/update-gensim-word2vec-model

# model = gensim.models.Word2Vec.load('pubmed.vocab_2.model')

# for corpus_tuple in archive_files:
#     (corpus_name, num_lines) = corpus_tuple
#     corpus_file = '{}{}.text'.format(base, corpus_name)
#     model.build_vocab(corpus_file=corpus_file, update=True, progress_per=1000)

# model.save('pubmed.vocab_2_1.model')

model = gensim.models.Word2Vec.load('pubmed.vocab_2_1.model')

for corpus_tuple in archive_files:
    (corpus_name, num_lines) = corpus_tuple
    corpus_file = '{}{}.text'.format(base, corpus_name)
    print("training", corpus_file)
    model.train(corpus_file=corpus_file, total_examples=num_lines, epochs=model.epochs, total_words=model.corpus_total_words)

model.save('pubmed_2_1.model')