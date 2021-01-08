# Looks for codes that most closely match the given word

"""
Given a single word vector.
That word vector is compared to summary description vectors to find the closest match
code descriptions /media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/datasets/code_descriptions
"""

import gensim
import json
import copy

model = gensim.models.Word2Vec.load('pubmed.model')

base_dir = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/datasets/code_descriptions'
file = 'dx_icd10.json'

codes_dict = {}

with open('{}/{}'.format(base_dir, file)) as json_file:
    code_rows = json.load(json_file)

    for code_row in code_rows:
        description = code_row['description']
        if description is not '' and description in model.wv.vocab:
            codes_dict[description] = code_row['codes']

codes_dict_items = codes_dict.items() 
descriptions = [description for description, codes in codes_dict_items]

def print_top(search_word):

    print('calculating')
    distances = model.wv.distances(search_word, descriptions)

    dists_and_codes = list(zip(distances, codes_dict_items))

    closest_desc_codes = sorted(dists_and_codes, key=lambda r: r[0], reverse=False)[0:12]

    for distance, desc_codes in closest_desc_codes:
        (description, codes) = desc_codes

        print("-----------------------------------------------------------------------------------")
        print(description, distance)
        print(codes)

    print("##################################################")


exit_me = False

while(exit_me == False):
    search_word = input('search word: ')
    if search_word == 'exit':
        exit_me = True
    else:
        print_top(search_word)