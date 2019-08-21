import os, numpy as np, pickle
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

modelname = "/lt/work/shiva/glove.840B.300d.txt"

input_file = "ProcessedData_8Jan.csv"

train = pd.read_csv(input_file, header=0, delimiter=",")
train['overview_text'] = train['overview_text'].str.lower()
train['overview_text'] = train['overview_text'].str.replace('http\S+|www.\S+', '', case=False)
train['overview_text'] = train['overview_text'].str.replace('-|\.|,|;|:', ' ', case=False)
train['overview_text'] = train['overview_text'].str.replace('\"|\'', '', case=False)

corpus = []
for p, i in zip(train['petition_id'], train['overview_text']):
    z = []
    for e in sent_tokenize(i.decode("utf-8")):
        z.append(e)
    corpus.append(z)
      
def getvocab(corpus):
    uniquewords = set()
    for x in corpus:
        for y in x:
            for w in word_tokenize(y.strip()):
                uniquewords.add(w)
    return uniquewords


vocab = getvocab(corpus)

embeddings = {}

lines = [line.strip().decode("utf-8") for line in open(modelname)]
for line in lines:
    row = line.split(" ")
    if row[0] in vocab:
        embeddings[row[0]] = [float(n) for n in row[1:]]
    
import pickle 
with open(r"glovered.pickle", "wb") as output_file:
    pickle.dump(embeddings, output_file)
        

