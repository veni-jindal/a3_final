# import required packages
# import tensorflow and keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense,  Dropout,  Bidirectional
from keras.regularizers import l2
# import train test split
from sklearn.model_selection import train_test_split
# import nltk and stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
# import numpy, pandas and pyplot
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import tokenizer and pad sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# packages to load the dataset  to data folder
from keras.utils import get_file
import tarfile
# glob
import glob
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import os
import pickle




# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

def extract_data():
	# path of data directory
	data_dir = get_file('aclImdb_v1.tar.gz', 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', cache_subdir = "datasets",hash_algorithm = "auto", extract = True, archive_format = "auto")
	# extracting the tarfile
	my_tar = tarfile.open(data_dir)
	# move extracted data to the data folder
	my_tar.extractall('./data/') 
	my_tar.close()

def preprocess(text):
  # customised stop words list with negative words removed from nltk stopword list
	stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
              'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
              'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 
              'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
              'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
              'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 
               'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',  'should', "should've", 'now', 'd', 'll', 
              'm', 'o', 're', 've', 'y']
	# lowercasing the given text
	text = text.lower()
	# removing the html tags from the text
	text = re.sub('<br\s*/><br\s*/>', '', text)
	# splitting the text to tokens
	tokens = text.split()
	# removing stopwords from the tokens list
	tokens = [word for word in tokens if word not in stop_words ]
	# removing all the special charcaters
	tokens = [re.sub('\W+','', token) for token in tokens]
	# initialising the lemmatizer from nltk 
	lem = WordNetLemmatizer()
	# lemmatizer on each token
	tokens = [lem.lemmatize(word) for word in tokens]
	# return text joined by space on tokens
	return ' '.join(tokens)


def embedding_mat(w2v, train_data, val_data):
    vocab_size_max = 50000
    embedding_dim = 100
    max_doc_len = 200
    # build tokenizer for text
    t = Tokenizer( num_words=vocab_size_max,filters='', lower=True, split=' ', oov_token= 1)  
    # fit tokenizer on train data
    t.fit_on_texts(train_data)
    # convert train, val, test data to encoding
    x_train = t.texts_to_sequences(train_data)
    x_val = t.texts_to_sequences(val_data)
    x_train = pad_sequences(x_train, maxlen=max_doc_len, truncating= 'post', padding= 'post')
    x_val = pad_sequences(x_val, maxlen=max_doc_len, truncating= 'post', padding= 'post')
    #x_test = t.texts_to_sequences(test_data)
    #x_test = pad_sequences(x_test, maxlen=max_doc_len, truncating= 'post', padding= 'post')
    # save tokenizer 
    if not os.path.exists('data'):
        os.mkdir('data')
    pickle.dump(t, open(os.path.join('data', "tokenizer.pkl"), "wb" ) )
    #vocab_size = len(vocab) + 1 (padding)
    vocab_size = len(t.word_index) + 1
    # construct embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in t.word_index.items():
        # initialise embedding vector with none
        embedding_vector = None
        # for unknown token (oov was set to 1 in tokeniser)
        if(i ==1):
            embedding_vector = np.random.randn(1,embedding_dim)
        # embedding vector fetch from the word2vec model
        else:
            embedding_vector = w2v[word]
        # if embedding vector not none add to embedding matrix
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    #print(embedding_matrix.shape)
    return embedding_matrix, x_train, x_val

def main():
	# extracting the data to data folder - one time step
	'''extract_data()'''
	

	# 1. load your training data
	# getting all the names of the text files in pos & neg folder
	txt_pos = glob.glob("./data/aclImdb/train/pos/*.txt")
	txt_neg = glob.glob("./data/aclImdb/train/neg/*.txt")
	# preparing the training data
	train = []
	# read the data from all files in pos directory
	for i in txt_pos:
		with open(i) as f:
			train.append(f.read())
	# read the data from all files in neg directory
	for i in txt_neg:
		with open(i) as f:
			train.append(f.read())
	# preparing list of pos labels (label = 1)
	labels_pos = [1]*len(txt_pos)
	# preparing list of neg labels (label = 0)
	labels_neg = [0]*len(txt_neg)
	# preparing list of labels combined
	labels = labels_pos + labels_neg
	
	# storing the data into a dataframe
	train_df = pd.DataFrame(columns = ['review', 'sentiment'])
	# storing text to review column
	train_df['review'] = train
	# storing labels to sentiment column
	train_df['sentiment'] = labels
	# shuffling the dataset
	train_df = shuffle(train_df)
	#applying preprocessing on the text 
	train_df['clean_text'] = train_df['review'].apply(lambda x: preprocess(x))
	#splitting into train and validation
	# getting values of the review texts
	X_text = train_df['clean_text'].values
	# getting labels
	y = np.ravel(train_df['sentiment'])
	# splitting into training and validation
	X_train_text, X_val_text , y_train, y_val = train_test_split(X_text, y, test_size = 0.2, random_state = 42)

	'''training word2vec model'''
	# getting tokens of training data
	txt = [text.split() for text in train_df['clean_text']]
	# learn word2vec embedding
	w2v = Word2Vec(txt, size=100, window=8, min_count=1, workers=4)
	w2v.save('models/w2v.model')
	'''preparing input data for neural network'''
	#loading the w2v model
	w2v = Word2Vec.load("models/w2v.model")
	# embedding matrix, X_train, X_val, X_test
	emb_mat, X_train, X_val = embedding_mat(w2v, X_train_text,X_val_text)

	# 2. Train your network
	# initialise the sequential network
	model = Sequential()
	# create an embedding layer for inputs
	model.add(Embedding(emb_mat.shape[0], emb_mat.shape[1], input_length=200, trainable = False, weights = [emb_mat]))
	# add LSTM layer having 100 units
	model.add(LSTM(100))
	# adding dense layer
	model.add(Dense(10, activation='sigmoid'))
	# output layer having sigmoid activation function
	model.add(Dense(1, activation='sigmoid'))
	# compile the model
	model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	# summary of the model
	model.summary()
	# fitting the model and storing hstory
	history = model.fit(X_train, y_train, epochs=25, batch_size= 128, validation_data= (X_val, y_val), verbose=2)
	
	# 3. Save your model
	model.save(os.path.join('models','20848879_NLP_model'))


	return


if __name__ == "__main__": 
	main()
	
	

	