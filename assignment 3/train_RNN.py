# import required packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Conv1D, MaxPooling1D, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical
# import train test split, numpy and pandas
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
# getting new data frame with lookback into 3 previous days 
def sequence(data, dates):
	features = data.columns[1:]
	print(features)
	# setting names of columns of dataframe
	names = []
	# loop over previous 3 days
	for j in range(1,4):
		# get all the 4 features for each day
		names += [f+'(t-{})'.format(j) for f in features]
	# add target column to the dataframe
	names.append('target(t)')
	# create a dataframe with column name list designed in above loop
	df_new = pd.DataFrame(columns= names, index= dates[:len(data)-3])
	
	# loop over all the days having 3 previous days
	for i in range(len(data)-3):
		# list to store values of each row (day)
		cols = []
	# loop for getting previous 3 days data
		for j in range(1,4):
			# getting every feature from each day
			for f in features:
				# adding value of jth previous day and f feature to cols list
				cols.append(data[f][i+j])
		# setting open price of current day as target variable
		cols.append(data[' Open'][i])
		# add the data to new dataframe
		df_new.loc[dates[i]]= cols
	# returning the new dataframe
	return(df_new)

# calculate rmse of data
def rmse(actual, pred):
  error = 0
  for i in range(len(actual)):
    diff = (actual[i] - pred[i])**2
    error += diff
  error = tf.math.sqrt(error/len(actual))
  return error

def main():
	'''# 1. load your training data'''
    # read the csv file
	data = pd.read_csv(os.path.join('data','q2_dataset.csv'))

	# getting date column
	dates = data['Date']
	# extract the 4 features volume, open, low and High
	features = [' Volume', ' Open', ' High', ' Low']
	# getting columns required to save data to csv file
	cols = ['Date',' Volume', ' Open', ' High', ' Low']
	# selecting only required column
	data = data[cols]
	# get data to save to csv
	data_csv = sequence(data, dates)
	# splitting the data into train and test data
	train, test = train_test_split(data_csv, test_size = 0.3, random_state = 42)
	# saving train data
	train.to_csv(os.path.join('data',"train_data_RNN.csv"))
	# saving test data
	test.to_csv(os.path.join('data',"test_data_RNN.csv"))

	'''loading training data & preprocessing'''
	# reading train data
	train_data = pd.read_csv(os.path.join('data','train_data_RNN.csv'))
	# column names of input features
	col_names = train_data.columns[1:-1]
	# getting input and output data
	X = train_data[col_names].values
	y = np.ravel(train_data['target(t)'])
	y = y.reshape(y.shape[0],1)
	# splitting the data in train & validation
	X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state = 21)
	# prepricessing
	scaler1 = MinMaxScaler()
	scaler2 = MinMaxScaler()
	# normalising the data
	X_train_norm = scaler1.fit_transform(X_train)
	y_train_norm = scaler2.fit_transform(y_train)
	# saving the two scalers into pkl file
	pickle.dump(scaler1, open(os.path.join('data', "scaler1.pkl"), "wb" ) )
	pickle.dump(scaler2, open(os.path.join('data', "scaler2.pkl"), "wb" ) )
	# transforming validation data
	X_val_norm = scaler1.transform(X_val)
	y_val_norm = scaler2.transform(y_val)
	# makinng inputs ready for LSTM 
	X_train_norm = X_train_norm.reshape(X_train_norm.shape[0],3,4)
	X_val_norm = X_val_norm.reshape(X_val_norm.shape[0], 3, 4)


	'''# 2. Train your network'''
	# initialise sequetial model
	model = Sequential()
	# add LSTM layer
	model.add(LSTM(units=70, input_shape = (X_train_norm.shape[1],X_train_norm.shape[2]) ))
	# add dense layer
	model.add(Dense(1))
	# compile model
	model.compile(optimizer='adam', loss='mean_squared_error')
	# fit model on train and validate
	history = model.fit(X_train_norm, y_train_norm, epochs=200, batch_size=8, validation_data = (X_val_norm, y_val_norm), verbose=1)
	'''predicting the output from training data and validation data'''
	y_train_pred = model.predict(X_train_norm)
	y_val_pred = model.predict(X_val_norm)
	'''inverse transform the predicted value to get value in actual scale'''
	y_train_pred = scaler2.inverse_transform(y_train_pred)
	y_val_pred = scaler2.inverse_transform(y_val_pred)
	print('RMSE loss on training data: {}'.format(rmse(y_train, y_train_pred)))
	print('RMSE loss on validation data: {}'.format(rmse(y_val, y_val_pred)))

	'''plot for training loop'''
	# plots for accuracy and loss for training
	fig, ax = plt.subplots(1,1, figsize=(7,4))
	# summarize history for loss
	plt.plot(tf.math.log(history.history['loss']))
	plt.plot(tf.math.log(history.history['val_loss']))
	plt.title('Log loss vs epoch with normalised data')
	plt.ylabel('log loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper right')
	fig.set_facecolor('white')
	plt.show()

	'''save your model'''
	model.save(os.path.join('models','20848879_RNN_model'))


	


if __name__ == "__main__":
    	main()
	