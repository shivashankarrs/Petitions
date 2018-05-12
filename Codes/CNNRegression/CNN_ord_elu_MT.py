
# coding: utf-8

# In[6]:

import string
import sys
import numpy as np
from six.moves import range
from six.moves import zip

if sys.version_info < (3,):
    maketrans = string.maketrans
else:
    maketrans = str.maketrans


def base_filter():
    f = string.punctuation
    f = f.replace("'", '')
    f += '\t\n'
    return f


def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    """Converts a text to a sequence of words (or tokens).
    # Arguments
        text: Input text (string).
        filters: Sequence of characters to filter out.
        lower: Whether to convert the input to lowercase.
        split: Sentence split marker (string).
    # Returns
        A list of words (or tokens).
    """
    if lower:
        text = text.lower()

    if sys.version_info < (3,) and isinstance(text, unicode):
        translate_map = dict((ord(c), unicode(split)) for c in filters)
    else:
        translate_map = maketrans(filters, split * len(filters))

    text = text.translate(translate_map)
    seq = text.split(split)
    return [i for i in seq if i]


def one_hot(text, n, filters=base_filter(), lower=True, split=" "):
    seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
    return [(abs(hash(w)) % (n - 1) + 1) for w in seq]


class Tokenizer1(object):
    def __init__(self, nb_words=None, filters=base_filter(),
                 lower=True, split=' ', char_level=False):
        '''The class allows to vectorize a text corpus, by turning each
        text into either a sequence of integers (each integer being the index
        of a token in a dictionary) or into a vector where the coefficient
        for each token could be binary, based on word count, based on tf-idf...

        # Arguments
            nb_words: the maximum number of words to keep, based
                on word frequency. Only the most common `nb_words` words will
                be kept.
            filters: a string where each element is a character that will be
                filtered from the texts. The default is all punctuation, plus
                tabs and line breaks, minus the `'` character.
            lower: boolean. Whether to convert the texts to lowercase.
            split: character or string to use for token splitting.
            char_level: if True, every character will be treated as a word.

        By default, all punctuation is removed, turning the texts into
        space-separated sequences of words
        (words maybe include the `'` character). These sequences are then
        split into lists of tokens. They will then be indexed or vectorized.

        `0` is a reserved index that won't be assigned to any word.
        '''
        self.word_counts = {}
        self.word_docs = {}
        self.filters = filters
        self.split = split
        self.lower = lower
        self.nb_words = nb_words
        self.document_count = 0
        self.char_level = char_level

    def fit_on_texts(self, texts):
        '''Required before using texts_to_sequences or texts_to_matrix

        # Arguments
            texts: can be a list of strings,
                or a generator of strings (for memory-efficiency)
        '''
        self.document_count = 0
        for text in texts:
            self.document_count += 1
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        #wcounts = list(self.word_counts.items())
        wcounts = self.word_counts.items()
        wcounts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in wcounts]
        # note that index 0 is reserved, never assigned to an existing word
        #self.word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))
        self.word_index = dict(zip(sorted_voc, range(1, len(sorted_voc) + 1)))

        self.index_docs = {}
        #for w, c in list(self.word_docs.items()):
        for w, c in self.word_docs.items():
            self.index_docs[self.word_index[w]] = c

    def fit_on_sequences(self, sequences):
        '''Required before using sequences_to_matrix
        (if fit_on_texts was never called)
        '''
        self.document_count = len(sequences)
        self.index_docs = {}
        for seq in sequences:
            seq = set(seq)
            for i in seq:
                if i not in self.index_docs:
                    self.index_docs[i] = 1
                else:
                    self.index_docs[i] += 1

    def texts_to_sequences(self, texts):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Returns a list of sequences.
        '''
        res = []
        for vect in self.texts_to_sequences_generator(texts):
            res.append(vect)
        return res

    def texts_to_sequences_generator(self, texts):
        '''Transforms each text in texts in a sequence of integers.
        Only top "nb_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.

        Yields individual sequences.

        # Arguments:
            texts: list of strings.
        '''
        nb_words = self.nb_words
        for text in texts:
            seq = text if self.char_level else text_to_word_sequence(text, self.filters, self.lower, self.split)
            vect = []
            for w in seq:
                i = self.word_index.get(w)
                if i is not None:
                    if nb_words and i >= nb_words:
                        continue
                    else:
                        vect.append(i)
            yield vect

    def texts_to_matrix(self, texts, mode='binary'):
        '''Convert a list of texts to a Numpy matrix,
        according to some vectorization mode.

        # Arguments:
            texts: list of strings.
            modes: one of "binary", "count", "tfidf", "freq"
        '''
        sequences = self.texts_to_sequences(texts)
        return self.sequences_to_matrix(sequences, mode=mode)

    def sequences_to_matrix(self, sequences, mode='binary'):
        '''Converts a list of sequences into a Numpy matrix,
        according to some vectorization mode.

        # Arguments:
            sequences: list of sequences
                (a sequence is a list of integer word indices).
            modes: one of "binary", "count", "tfidf", "freq"
        '''
        if not self.nb_words:
            if self.word_index:
                nb_words = len(self.word_index) + 1
            else:
                raise Exception('Specify a dimension (nb_words argument), '
                                'or fit on some text data first.')
        else:
            nb_words = self.nb_words

        if mode == 'tfidf' and not self.document_count:
            raise Exception('Fit the Tokenizer on some data '
                            'before using tfidf mode.')

        X = np.zeros((len(sequences), nb_words))
        for i, seq in enumerate(sequences):
            if not seq:
                continue
            counts = {}
            for j in seq:
                if j >= nb_words:
                    continue
                if j not in counts:
                    counts[j] = 1.
                else:
                    counts[j] += 1
            for j, c in list(counts.items()):
                if mode == 'count':
                    X[i][j] = c
                elif mode == 'freq':
                    X[i][j] = c / len(seq)
                elif mode == 'binary':
                    X[i][j] = 1
                elif mode == 'tfidf':
                    tf = 1 + np.log(c)
                    idf = np.log(1 + self.document_count / (1 + self.index_docs.get(j, 0)))
                    X[i][j] = tf * idf
                else:
                    raise Exception('Unknown vectorization mode: ' + str(mode))
        return X


# In[8]:

import os, numpy as np, pickle
os.chdir("Petitions/") #Path to petitions
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'

modelname = "glovered.pickle"
with open(modelname, 'rb') as handle:
        embeddings = pickle.load(handle)


import pandas as pd

#Petitions are sorted by date

petitions = pd.read_csv("USPetitions_7K.csv", header=0)
redpet = petitions[petitions['signs']<>'null']
redpet['signs'] = redpet['signs'].astype('int')
redpet = redpet[redpet['signs']>=150]

trainpetitions = redpet.iloc[0:818]
valpetitions = redpet.iloc[818:920]
testpetitions = redpet.iloc[920:]
trainpetitions['fulltext'] = trainpetitions['title'].astype('str')+' '+trainpetitions['content'].astype('str')
valpetitions['fulltext'] = valpetitions['title'].astype('str')+' '+valpetitions['content'].astype('str')
testpetitions['fulltext'] = testpetitions['title'].astype('str')+' '+testpetitions['content'].astype('str')
 

from nltk import word_tokenize, sent_tokenize

import math

#encoding will change for UK, as there are 2 more classes (10, 100)

def getsegment(val):
    if math.log(val) > math.log(100000):
        return [1, 1, 1]
    if math.log(val) > math.log(10000):
        return [1, 1, 0]
    if math.log(val) > math.log(1000):
        return [1, 0, 0]
    else: 
        return [0, 0, 0]

train_labels = np.array(trainpetitions['signs'])
val_labels = np.array(valpetitions['signs'])
test_labels = np.array(testpetitions['signs'])

trainc_labels =  trainpetitions['signs'].apply(getsegment)
valc_labels = valpetitions['signs'].apply(getsegment)
testc_labels = testpetitions['signs'].apply(getsegment)

trainc_labels = trainc_labels.reset_index()
valc_labels = valc_labels.reset_index()
testc_labels = testc_labels.reset_index()

traino_labels = np.array(trainc_labels['signs'])
valo_labels = np.array(valc_labels['signs'])
testo_labels = np.array(testc_labels['signs'])

from keras.preprocessing.sequence import pad_sequences

def gettraining(trainleft, tokenizer):
    MAX_SEQUENCE_LENGTH = 90        
    return pad_sequences(tokenizer.texts_to_sequences(trainleft), maxlen = MAX_SEQUENCE_LENGTH)        
    
import pandas as pd, numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

train = pd.read_csv("USPetitions_7K.csv", header=0, delimiter=",")
train['text']  = train['text'].str.lower()
train['text'] = train['text'].str.replace('http\S+|www.\S+', '', case=False)
train['text'] = train['text'].str.replace('-|\.|,|;|:', ' ', case=False)
train['text'] = train['text'].str.replace('\"|\'', '', case=False)

corpus = []
for p, i in zip(train['id'], train['text']):
    z = []
    for e in sent_tokenize(i.decode("utf-8")):
        z.append(e)
    corpus.append(z)
      
def gettokenizer(corpus):
    MAX_NB_WORDS = 100000
    fulldata=[]
    for j, mani in enumerate(corpus):
        for k, sent in enumerate(mani):
            fulldata.append(sent)
    tot = np.array(fulldata)
    tokenizer = Tokenizer1(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(tot)
    return tokenizer

tokenizer = gettokenizer(corpus)

def cleantext(x):
    x['cleandata'] = x['fulltext'].replace('\d+', 'NUM ', regex=True)
    return x
    
trainpetitions = cleantext(trainpetitions)
valpetititons = cleantext(valpetitions)
testpetitions = cleantext(testpetitions)

trdata = gettraining(trainpetitions['cleandata'].tolist(), tokenizer)
valdata = gettraining(valpetitions['cleandata'].tolist(), tokenizer)
tedata = gettraining(testpetitions['cleandata'].tolist(), tokenizer)

#load custom features

ctrain = np.loadtxt("trainscustomwtp")
cval = np.loadtxt("valscustomwtp")
ctest = np.loadtxt("testscustomwtp")


def predict(rpred):
    pred = []
    for i in rpred:
        if i > 0.5:
           pred.append(1)
        else:
           pred.append(0)

    return np.array(pred)


def lpredict(rpred):
    pred = []
    for i in rpred:
	if i>0:
	        if math.log(i) > math.log(100000):
        	   pred.append(1)
		else:
		   pred.append(0)
        else:
           pred.append(0)

    return np.array(pred)

def getlabels(labels, x):
    vlabels = []
    for i in labels:
        vlabels.append(i[x])
    return vlabels

# This is binary classification for US petitions, must be 3-class for UK petitions
def evaluate(cnnmodel, tedata, ctest, testo_labels):
    from sklearn.metrics import f1_score
    rpred = np.array(cnnmodel.predict([tedata, ctest], batch_size=128, verbose=0)[0])
    
    pred = lpredict(rpred)
    print pred.shape, np.sum(pred)
    testl = []
    for i in testo_labels:
        if np.sum(i)>2:    
        	testl.append(1)
	else:
		testl.append(0)

    print len(testl), np.sum(np.array(testl)), "check"

    return f1_score(pred, testl,average='macro')

import math
from keras import backend as K

def cnnmodel(x_train, c_train, y_train, x_val, c_val, y_val, tokenizer, o_train, o_val, tedata, ctest, testo_labels):

    from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D, Embedding, AveragePooling1D, TimeDistributed, GlobalMaxPooling1D, Merge
    from keras.preprocessing.text import Tokenizer
    from keras.models import Model
    import pandas as pd
    import keras

    MAX_SEQUENCE_LENGTH = 90
    MAX_NB_WORDS = 100000
    
    EMBEDDING_DIM = 300
    HIDDEN_DIMS = 5

    from keras.layers import Dense, Input, Flatten

    word_index = tokenizer.word_index
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.normal(-4.2, 4.2, 300)

    embedding_layer = Embedding(nb_words + 1, EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)

    print('Training model.')
    FILTERS = 3
    
    review_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(review_input)

    custom = Input(shape=(20,), dtype='float32')

    cnn = Conv1D(FILTERS, 1, padding='valid', activation='relu', strides=1) (embedded_sequences)
    cnn = GlobalMaxPooling1D()(cnn)

    cnn1 = Conv1D(FILTERS, 2, padding='valid', activation='relu', strides=1) (embedded_sequences)
    cnn1 = GlobalMaxPooling1D()(cnn1)

    cnn2 = Conv1D(FILTERS, 3, padding='valid', activation='relu', strides=1) (embedded_sequences)
    cnn2 = GlobalMaxPooling1D()(cnn2)

    cnnembed = keras.layers.concatenate([cnn, cnn1, cnn2])
    
    customembed = Dense(1, activation='tanh')(custom)

    cnnembed = Dense(HIDDEN_DIMS, activation='tanh')(cnnembed)

    cnnembedc = keras.layers.concatenate([customembed, cnnembed])
    cnnembedc = Dense(2, activation='tanh')(cnnembedc)
   
    label1 = np.array(getlabels(o_train,0))
    label2 = np.array(getlabels(o_train,1))
    label3 = np.array(getlabels(o_train,2))

    vlabel1 = np.array(getlabels(o_val,0))
    vlabel2 = np.array(getlabels(o_val,1))
    vlabel3 = np.array(getlabels(o_val,2))

    sign = Dense(1, activation='linear')(cnnembedc)

    l1 = Dense(1, activation='sigmoid')(cnnembed)
    l2 = Dense(1, activation='sigmoid')(cnnembed)
    l3 = Dense(1, activation='sigmoid')(cnnembed)

    big_model = Model([review_input, custom],  output = [sign, l1, l2, l3])

    dummy_model = Model(review_input, output = [cnnembed])

    dummy_model.trainable = False

    big_model.compile(loss=['mean_squared_logarithmic_error','binary_crossentropy', 'binary_crossentropy','binary_crossentropy'], optimizer='Adam', metrics=['mean_squared_logarithmic_error','binary_crossentropy', 'binary_crossentropy','binary_crossentropy'], loss_weights=[1,2,2,2])

    #Iterations set using early stopping ob validation set performance

    iterations = 7500
    for i in range(iterations):
        big_model.fit([x_train, c_train], [y_train, label1, label2, label3], batch_size=32, epochs=1, validation_data=([x_val, c_val], [y_val, vlabel1, vlabel2, vlabel3]))

	pred = np.array(big_model.predict([tedata, ctest], batch_size=128, verbose=0)[0])

	import keras, math
	y_classes = []
	for v, i in enumerate(pred):
    	    if i>0:
            	y_classes.append(math.log(i))
	    else:
        	y_classes.append(0)


	from sklearn.metrics import mean_squared_error, mean_absolute_error
	from scipy.stats import pearsonr
	cumerr = []
	true_classes = []
	for v, i in enumerate(testpetitions['signs']):
            true_classes.append(math.log(i))
            cumerr.append((math.fabs(y_classes[v]-math.log(i))*100.0)/(math.log(i)))

        print mean_absolute_error(np.array(true_classes), np.array(y_classes))
        print "mape", np.average(np.array(cumerr))
        print "macro 0/1",  evaluate(big_model, tedata, ctest, testo_labels)
    return big_model

combmodel = cnnmodel(trdata, ctrain, train_labels, valdata, cval, val_labels, tokenizer, traino_labels, valo_labels, tedata, ctest, testo_labels)



