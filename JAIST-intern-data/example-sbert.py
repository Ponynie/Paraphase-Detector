from sentence_transformers import SentenceTransformer

# load model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# list of example sentences
sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']

# get sentence embedding
sentence_embeddings = model.encode(sentences)

# print out
for sentence, embedding in zip(sentences, sentence_embeddings):
    print("Sentence:", sentence)
    #print("Embedding:", embedding)
    print("Embedding(first 10 values): "+" ".join(list(map(str,embedding[0:10]))))
    print("")
