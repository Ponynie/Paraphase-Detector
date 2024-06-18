#!/usr/local/bin/python3

import spacy

# Load the language model
nlp = spacy.load("en_core_web_sm")

sentence = 'Tom took a picture of Mt. Fuji.'

# nlp function returns an object with individual token information, 
# linguistic features and relationships
doc = nlp(sentence)

for token in doc:
    word = token.text
    dependency_type = token.dep_
    head_word = token.head.text
    print("{:s} =({:s})=> {:s}".format(word,dependency_type,head_word))
