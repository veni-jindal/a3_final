# import required packages
# import tensorflow and keras
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

# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow
# calculate rmse of data
def rmse(actual, pred):
  error = 0
  for i in range(len(actual)):
    diff = (actual[i] - pred[i])**2
    error += diff
  error = tf.math.sqrt(error/len(actual))
  return error

def main():
	# 1. Load your saved model

	# 2. Load your testing data
	test_data = pd.read_csv(os.path.join('data','test_data_RNN.csv'))
	# getting the columns of input features and target values 
	test_cols = test_data.columns[1:-1]
	# Convert date to datetime
	test_data['Date'] = pd.to_datetime(test_data['Date'])
	# sort values by date
	test_data = test_data.sort_values(by='Date')
	# getting dates in list
	dates = test_data['Date']
	''''loading the scalers used while training'''
	with open(os.path.join('data', 'scaler1.pkl'), 'rb') as f:
		scaler1= pickle.load(f)
	with open(os.path.join('data', 'scaler2.pkl'), 'rb') as f:
		scaler2 = pickle.load(f)

	'''normalising the test data'''
	# array normalised
	X_test_norm = scaler1.transform(test_data[test_cols].values)
	# getting target column from the dataframe
	y_test = np.ravel(test_data['target(t)'])
	# transform the target to normalised value
	y_test_norm = scaler2.transform(y_test.reshape(y_test.shape[0],1))

	'''3. Run prediction on the test data and output required plot and loss'''
	# reshaping X_test to make it ready for LSTM
	X_test_norm = X_test_norm.reshape(X_test_norm.shape[0],3,4)
	# load saved model
	model = tf.keras.models.load_model('models/20848879_RNN_model')
	# predict the target variable from model
	y_test_pred = model.predict(X_test_norm)
	# converting y_test_pred to unnormalised values
	y_pred_test_un = scaler2.inverse_transform(y_test_pred)

	# find RMSE of the test data predictions
	print('RMSE loss on test data: {}'.format(rmse(y_test, y_pred_test_un)))

	# plot the actual vs predicted values
	fig = plt.figure(figsize= (10, 7))
	plt.plot_date(dates, y_test, linestyle = 'solid', marker = None, color = 'red', label = 'Actual')
	plt.plot_date(dates, y_pred_test_un,linestyle = 'solid', marker = None, color = 'green', label = 'Predicted' )
	plt.title("Opening price of stocks sold")
	plt.xlabel("Time")
	plt.ylabel("Stock Opening Price")
	plt.legend()
	fig.set_facecolor('white')
	plt.show()




if __name__ == "__main__":
	main()
	

	

	