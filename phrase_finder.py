# Looks for codes that most closely match the given word

"""
Given a phrase, that phrase is used to create a summary vector.
The phrase summary vector is compared to summary description vectors to find the closest match
code descriptions /media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/datasets/code_descriptions
"""

import gensim
import json
import copy
import numpy as np
from nltk.corpus import stopwords


model = gensim.models.Word2Vec.load('pubmed.model')

base_dir = '/media/bworkman/4facacdc-e765-4238-b04e-15dba16122c9/datasets/code_descriptions'
file = 'dx_icd10_ref.json'

codes_dict = {}

stopwords = stopwords.words('english')

def sentence_to_array(sentence):
    return [w for w in sentence.lower().split() if w not in stopwords]

with open('{}/{}'.format(base_dir, file)) as json_file:
    code_rows = json.load(json_file)

    for code_row in code_rows:

        code = code_row['diagnosis_code']
        description = code_row['long_desc']
        
        codes_dict[code] = sentence_to_array(description)

codes_dict_items = codes_dict.items() 
descriptions = [description for code, description in codes_dict_items]

def print_top(search_sentence):

    print('calculating')
    search_sentence_arr = sentence_to_array(search_sentence)

    distances = []

    for description in descriptions:
        wm_distance = model.wmdistance(search_sentence_arr, description)
        distances.append(wm_distance)

    dists_and_codes = list(zip(distances, codes_dict_items))

    closest_desc_codes = sorted(dists_and_codes, key=lambda r: r[0], reverse=False)[0:12]

    for distance, code_desc in closest_desc_codes:
        (code, description) = code_desc

        print("-----------------------------------------------------------------------------------")
        print(distance, code, description)

    print("##################################################")


exit_me = False

while(exit_me == False):
    search_description = input('search description: ')
    if search_description == 'exit':
        exit_me = True
    else:
        print_top(search_description)
