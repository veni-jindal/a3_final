# import required packages
import glob
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
from sklearn.utils import shuffle
from nltk.stem import WordNetLemmatizer
import os
import pickle

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow

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

def accuracy(pred, actual):
  cor = 0
  for i in range(len(pred)):
    if(pred[i] == actual[i]):
      cor +=1
  return (cor/len(actual))

def main():
	# 1. Load your saved model
	model = tf.keras.models.load_model('models/20848879_NLP_model')

	#2. Load your testing data
	# getting list of file names inside the folder
	test_pos = glob.glob("./data/aclImdb/test/pos/*.txt")
	test_neg = glob.glob("./data/aclImdb/test/neg/*.txt")
	# preparing the test data
	test = []
	# read the data from all files in pos directory
	for i in test_pos:
		with open(i) as f:
			test.append(f.read())
	# read the data from all files in neg directory
	for i in test_neg:
		with open(i) as f:
			test.append(f.read())
	# preparing list of pos labels (label = 1)
	test_labels_pos = [1]*len(test_pos)
	# preparing list of neg labels (label = 0)
	test_labels_neg = [0]*len(test_neg)
	# preparing list of labels combined
	test_labels = test_labels_pos + test_labels_neg
	# storing the data into a dataframe
	test_df = pd.DataFrame(columns = ['review', 'sentiment'])
	# storing text to review column
	test_df['review'] = test
	# storing labels to sentiment column
	test_df['sentiment'] = test_labels
	#applying preprocessing on the text 
	test_df['clean_text'] = test_df['review'].apply(lambda x: preprocess(x))
	# load the tokeniser fitted on train data
	with open(os.path.join('data', 'tokenizer.pkl'), 'rb') as f:
		t = pickle.load(f)
	# maximum sentence length
	max_doc_len = 200
	# converting text to integer encoding
	x_test = t.texts_to_sequences(test_df['clean_text'].values)
	# padding sequences
	X_test = pad_sequences(x_test, maxlen=max_doc_len, truncating= 'post', padding= 'post')
	#Prediction on test set
	y_pred = model.predict_classes(X_test)
	# getting true labels
	y_test = np.ravel(test_df['sentiment'])
	# calculate accuracy
	print("Testing accuracy is {}".format(accuracy(y_pred, y_test)))

	return




if __name__ == "__main__": 
	main()
	

	# 2. Load your testing data

	# 3. Run prediction on the test data and print the test accuracy