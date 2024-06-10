from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 

stemmer = PorterStemmer()
stopword_list = set(stopwords.words('english'))

sentence = "It is fine today. Today's weather is fine. Tom likes rainy weather but Sue not. Jane likes dogs and Mike likes cat."
#sentence = "All work and no play makes jack dull boy."

tokenized_words = word_tokenize(sentence)
stemmed_words = []
for w in tokenized_words:
    stemmed_words.append( stemmer.stem(w.lower()) )

words = []
for w in stemmed_words:
    if w not in stopword_list:
        words.append(w)
print(words)
