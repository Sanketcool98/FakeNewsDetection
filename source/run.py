# importing the requires Libraries.
import os.path
import pickle
import sys
from scipy.sparse import csr_matrix
sys.path.append(os.path.join(os.path.dirname('C:/Users/Owner/Desktop/Project/source/Driver.ipynb')))
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from Visualize import *
from LRM import ModelLogisticRegression
from NB import NaiveBayes
import Evaluate
# import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
MODEL = {1: "Logistic Regression Model", 2: "Naive Bayes Model"}
PARAMS = {1: (0.0004, 0.0003, 1000), 2: 2}
TITLE = {"ModelLogisticRegression": "Logistic Regression Model",
		 "NaiveBayes": "Naive Bayes Model",
		 "GridSearchCV": "Support Vector Machine Model",
		 "RandomForestClassifier": "Random Forest Model"}

def fetchDataLRM(fileName, modelObj):
	data = pd.read_csv('C:/Users/Owner/Desktop/Project/Datasets/Data.csv')
	# print(data.info())
	print("Enter the size of data to train and test (max data - 20000): ")
	dataSize = int(input())
	data = data[:dataSize]
	trainDataSize = int(abs(dataSize * 0.8))
	testStartIndex = trainDataSize
	testEndIndex = dataSize
	''' fetching data text feature from data set for training '''
	X_train = data.iloc[:trainDataSize, 2].values
	''' fetching real or fake  feature from data set for training '''
	y_train = data.iloc[:trainDataSize, -1].values
	''' fetching data text feature from data set for testing  '''
	X_test = data.iloc[testStartIndex:testEndIndex, 2].values
	# print(X_test)
	''' fetching data text feature from data set for testing '''
	y_test = data.iloc[testStartIndex:testEndIndex, -1].values
	print("The data split is as follows:")
	print("X-train :", len(X_train))
	print("Y-train :", len(y_train))
	print("X-test :", len(X_test))
	print("Y-test :", len(y_test))
	stopwords_ = [word.encode('utf-8')for word in list(stopwords.words('english'))]
	'Optimization of feature generation based on Model'
	maxFeatures = 50000
	tfidf = TfidfVectorizer(min_df=1, max_features=maxFeatures, stop_words=stopwords_)
	text = tfidf.fit(data.text)
	x = text.transform(data.text)
	return X_train, y_train, X_test, y_test
def fetchDataNB(fileName, modelObj):
	data = pd.read_csv('C:/Users/Owner/Desktop/Project/Datasets/Data.csv')
	# print(data.info())
	print("Enter the size of data to train and test (max data - 20000): ")
	dataSize = int(input())
	data = data[:dataSize]
	trainDataSize = int(abs(dataSize * 0.8))
	testStartIndex = trainDataSize
	testEndIndex = dataSize
	''' fetching data text feature from data set for training '''
	X_train = data.iloc[:trainDataSize, 1].values
	''' fetching real or fake  feature from data set for training '''
	y_train = data.iloc[:trainDataSize, -1].values
	''' fetching data text feature from data set for testing  '''
	X_test = data.iloc[testStartIndex:testEndIndex, 1].values
	# print(X_test)
	''' fetching data text feature from data set for testing '''
	y_test = data.iloc[testStartIndex:testEndIndex, -1].values
	print("The data split is as follows:")
	print("X-train :", len(X_train))
	print("Y-train :", len(y_train))
	print("X-test :", len(X_test))
	print("Y-test :", len(y_test))
	stopwords_ = [word.encode('utf-8')for word in list(stopwords.words('english'))]
	'Optimization of feature generation based on Model'
	maxFeatures = 50000
	tfidf = TfidfVectorizer(min_df=1, max_features=maxFeatures, stop_words=stopwords_)
	text = tfidf.fit(data.text)
	x = text.transform(data.text)
	return X_train, y_train, X_test, y_test
def fetchDataRF(fileName, modelObj):
	data = pd.read_csv('C:/Users/Owner/Desktop/Project/Datasets/Data.csv')
	stopwords_ = [word.encode('utf-8')for word in list(stopwords.words('english'))]
	'Optimization of feature generation based on Model'
	maxFeatures = 50000
	X = data.text.values
	y = data.label.values
	tfidf = TfidfVectorizer(min_df=1, max_features=maxFeatures, stop_words=stopwords_)
	text = tfidf.fit(X)
	x = text.transform(X)
	X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=0.25, stratify=y)
	return X_train, y_train, X_test, y_test
def runModelRF(modelObj):
	print("Enter the file path of the data set to be used: (currently hard coded)")
	# fileName = input()
	''' fetch the data split '''
	X_train, y_train, X_test, y_test = fetchDataRF('C:/Users/Owner/Desktop/Project/Datasets/Data.csv', modelObj)
	print("The data split is as follows:")
	print("X-train :", X_train.shape)
	print("Y-train :", len(y_train))
	print("X-test :", X_test.shape)
	print("Y-test :", len(y_test))
	''' fit the Train data '''
	modelObj.fit(X_train, y_train)
	''' predict using test data '''
	pred = modelObj.predict(X_test)
	writeValsToPickleFile(pred, 'Prediction-' + modelObj.__class__.__name__)
	writeValsToPickleFile(y_test, 'Actual_data-' + modelObj.__class__.__name__)
	print("\nEvaluation on test data:\n")
	''' Evaluation of testing data and prediction : based on accuracy, precision , recall of the data  '''
	mapResults = Evaluate.precision_recall_evaluation(pred, y_test)
	mapResults['Accuracy'] = Evaluate.accuracy(pred, y_test)
	print('\n Writing the result to a text file for reference')
	writeResultsToTextFile(mapResults, TITLE[modelObj.__class__.__name__])
	print("\nVisualization of the output\n")
	''' Visualize the output '''
	plotScatterGraphForPrediction(pred, y_test, modelObj.__class__.__name__)
	if modelObj.__class__.__name__ == 'ModelLogisticRegression':
		loss_array = modelObj.loss_array
		writeValsToPickleFile(loss_array, 'loss_data-' + modelObj.__class__.__name__)
		loss_vs_iteration_plot(loss_array)
	sys.exit(0)
def runModelLRM(modelObj):
	print("Enter the file path of the data set to be used: (currently hard coded)")
	# fileName = input()
	''' fetch the data split '''
	X_train, y_train, X_test, y_test = fetchDataLRM('C:/Users/Owner/Desktop/Project/Datasets/Data.csv', modelObj)

	''' fit the Train data '''
	modelObj.fit(X_train, y_train)
	''' predict using test data '''
	pred = modelObj.predict(X_test)
	writeValsToPickleFile(pred, 'Prediction-'+modelObj.__class__.__name__)
	writeValsToPickleFile(y_test, 'Actual_data-'+modelObj.__class__.__name__)
	print("\nEvaluation on test data:\n")
	''' Evaluation of testing data and prediction : based on accuracy, precision , recall of the data  '''
	mapResults = Evaluate.precision_recall_evaluation(pred, y_test)
	mapResults['Accuracy'] = Evaluate.accuracy(pred, y_test)
	print('\n Writing the result to a text file for reference')
	writeResultsToTextFile(mapResults, TITLE[modelObj.__class__.__name__])
	print("\nVisualization of the output\n")
	''' Visualize the output '''
	plotScatterGraphForPrediction(pred, y_test, modelObj.__class__.__name__)
	if modelObj.__class__.__name__ == 'ModelLogisticRegression':
		loss_array = modelObj.loss_array
		writeValsToPickleFile(loss_array, 'loss_data-'+modelObj.__class__.__name__)
		loss_vs_iteration_plot(loss_array)
	sys.exit(0)
	
def runModelNB(modelObj):
		print("Enter the file path of the data set to be used: (currently hard coded)")
		# fileName = input()
		''' fetch the data split '''
		X_train, y_train, X_test, y_test = fetchDataLRM('C:/Users/Owner/Desktop/Project/Datasets/Data.csv', modelObj)
		
		''' fit the Train data '''
		modelObj.fit(X_train, y_train)
		''' predict using test data '''
		pred = modelObj.predict(X_test)
		writeValsToPickleFile(pred, 'Prediction-' + modelObj.__class__.__name__)
		writeValsToPickleFile(y_test, 'Actual_data-' + modelObj.__class__.__name__)
		print("\nEvaluation on test data:\n")
		''' Evaluation of testing data and prediction : based on accuracy, precision , recall of the data  '''
		mapResults = Evaluate.precision_recall_evaluation(pred, y_test)
		mapResults['Accuracy'] = Evaluate.accuracy(pred, y_test)
		print('\n Writing the result to a text file for reference')
		writeResultsToTextFile(mapResults, TITLE[modelObj.__class__.__name__])
		print("\nVisualization of the output\n")
		''' Visualize the output '''
		plotScatterGraphForPrediction(pred, y_test, modelObj.__class__.__name__)
		if modelObj.__class__.__name__ == 'ModelLogisticRegression':
			loss_array = modelObj.loss_array
			writeValsToPickleFile(loss_array, 'loss_data-' + modelObj.__class__.__name__)
			loss_vs_iteration_plot(loss_array)
		sys.exit(0)
def writeResultsToTextFile(mapResults, model):
	fname = "../EvaluationReports/"+model+'_Evaluation_Report'+'.txt'
	if os.path.exists(fname):
		os.remove(fname)
	fileModel = open(fname, 'w')
	topic = " Evaluation Report of "+model+" "
	hashLen = 90-len(topic)
	hashLen = int(hashLen/2)
	print(hashLen)
	filler = "#" * hashLen + topic + "#" * hashLen
	if len(filler) < 90:
		filler += "#"
	fileModel.write("#" * 90 + "\n" + filler + "\n" + "#" * 90 + "\n\n")
	for results in mapResults:
		fileModel.write(results + ": " + str(mapResults[results]) + "\n")
	fileModel.close()
def selectTasks():
	while True:
		print("\nSelect the Model for classification:")
		print("Enter 1 : Logistic Regression")
		print("Enter 2 : Naive Bayes")
		print("Enter 3 : Random Forest Model using Sklearn library")
		print("Enter 4 : To exit!!!!")
		options = {1: ModelLogisticRegression,
				   2: NaiveBayes,
				   3: RandomForestClassifier}
		print("Enter Your Choice >>> ")
		x = int(input())
		if x == 4:
			sys.exit(0)
		elif x == 3:
			print("Classification on Random Forest Model using SKLearn Library")
			runModelRF(options[x](n_jobs=2, random_state=0))
			break
		elif x == 1:
			print("Classification on LRM")
			runModelLRM(options[x](PARAMS[x]))
			break
		else:
			print("Classification on NB")
			runModelLRM(options[x](PARAMS[x]))

def writeValsToPickleFile(data, name):
	fName = '../PickleFilesForActualAndPredicted/'+name+'.pickle'
	if os.path.exists(fName):
		os.remove(fName)
	fileIndex = open(fName, 'wb')
	pickle.dump(data, fileIndex)
	fileIndex.close()


if __name__ == '__main__':
	print("Welcome to Fake News Detection")
	selectTasks()
