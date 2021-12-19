import numpy as np
from scipy.sparse import csr_matrix
# Logistic Regression Model class
class ModelLogisticRegression:
	# init method
	def __init__(self, params):     # learning_rate,threshold_tolerance=0.005,maximum_iterations=1000):<br>
		self.loss_array = []
		self.maximum_iterations = params[2]     # maximum_iterations<br>
		self.threshold_tolerance = params[1]    # threshold_tolerance<br>
		self.learning_rate = params[0]  # learning_rate<br>
	# train the model.
	
	def fit(self, X, y):
		iterations = 1
		self.weight = np.array(len(X))
		loss_difference = float('inf')
		loss = self.calculatelogisticLoss(X, y)
		while iterations < self.maximum_iterations and loss_difference > self.threshold_tolerance:
			iterations = iterations + 1
			scores = []
			for x in X:
				scores.append(np.dot(self.weight.T, x))
			error = sigmoid(scores) - y
			gradient = np.dot(X.T, error)
			self.weight = self.weight - (self.learning_rate * gradient)
			loss = self.calculatelogisticLoss(X, y)
			loss_difference = abs(loss - self.calculatelogisticLoss(X, y))
			self.loss_array.append(loss)
	# calculation of logistic Loss
	
	def calculatelogisticLoss(self, X, y):
		res = []
		for x in X:
			res.append(np.dot(self.weight.T, x))
		return -1 * (np.sum((y * np.log(sigmoid(res))) + ((1 - y) * np.log(1 - sigmoid(res)))))
		# test the model
	
	def predict(self, X):
		res = []
		for x in X:
			res.append(np.dot(self.weight.T, x))
		return np.round(sigmoid(res))

# compute the sigmoid
def sigmoid(scores):
	res = []
	for score in scores:
		res.append(1.0 / float(1 + np.exp(-score)))
	return np.array(res)
