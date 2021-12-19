import math
import sys
import numpy
import numpy as np

class NaiveBayes:
	from scipy.sparse import csr_matrix
	
	def __init__(self, pp):
		print("Naive Bayes Classifier")
	
	# initialize the model
	
	def fit(self, X, y):
		self.X = X[:]
		self.y = y
		self.docCount = len(X)
		self.vocabularyCount = len(X)
		self.categories, self.categoryCount = self.createFeatureDictionary()
		self.featureCount = {}
		for classes in self.categories:
			self.featureCount[classes] = len(self.categories[classes])
		self.train()
	
	def createFeatureDictionary(self):
		# create dictionary for each class
		# key : class , value = {feature : frequency}
		categories = {}
		categoryCount = {}
		for y in np.unique(self.y):
			categories[y] = {}
			categoryCount[y] = len(self.y[self.y == y])
		for j in range(self.docCount):
			for i in range(self.vocabularyCount):
				if (self.X[j] | self.X[i]) != 0:
					if i not in categories[self.y[j]]:
						categories[self.y[j]][i] = 1
					else:
						categories[self.y[j]][i] += 1
				else:
					categories[self.y[j]][i] = 0
				return categories, categoryCount
		
	def train(self):	
		# train the model
		# calculate the prior probabilities and conditional probabilities '''
		self.priorProbab = {}
		self.conditionalProbab = {}
		for classes in self.categories:
			self.priorProbab[classes] = math.log(self.featureCount[classes] + 1 / self.docCount)
			self.conditionalProbab[classes] = {}
			for features in self.categories[classes]:
				self.conditionalProbab[classes][features] = math.log((self.categories[classes][features] + 1) / (
					float(self.featureCount[classes] + self.vocabularyCount)))
		
	def predict(self, X_test):
		pred = []
		for x in X_test:
			pred.append(self.check(x))	
		return pred
	
	def check(self, x):
		# test the data
		docfeatures = []
		for i in range(len(str(x))):
			if x != 0:
				docfeatures.append(x)
		val = {}
		for classes in self.categories:
			val[classes] = self.priorProbab[classes]
		unseenProbab = {}
		for classes in self.categories:
			unseenProbab[classes] = math.log(1 / float(self.featureCount[classes] + self.vocabularyCount))
		for classes in self.categories:
			for feature in docfeatures:
				if feature in self.categories[classes]:
					val[classes] += self.categories[classes][feature]
			else:
				val[classes] += unseenProbab[classes] 
		sortedMap = sorted(val.items(), reverse=True)
		return sortedMap[0][0]
		

if __name__ == '__main__':
	# testing Naive Bayes accuracy
	n = NaiveBayes(1)
