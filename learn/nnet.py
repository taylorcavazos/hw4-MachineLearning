# Implement class to create a neural net object with variable number of input, hidden, and output layers
import numpy as np
from .sigmoid import *

class NNET(object):
	def __init__(self, num_input, num_hidden, num_output):
		"""
		Create neural net with specified number of input, hidden, and output neurons
		"""
		self.input = num_input # number input neurons
		self.hidden = num_hidden # number hidden neurons
		self.output = num_output # number output neurons
		### initialize random weights
		self.wh = np.random.randn(self.input, self.hidden)*np.sqrt(2.0/self.input)
		self.wo = np.random.randn(self.hidden, self.output)*np.sqrt(2.0/self.input)
		### initialize biases to small random values
		self.bh = np.random.uniform(size=(1, self.hidden)) # bias for hidden layer
		self.bo = np.random.uniform(size=(1, self.output)) # bias for output layer
		### initalize activations
		self.ah = np.ones(self.hidden) # activation for hidden layer
		self.ao = np.ones(self.output) # activation for output layer
		### allow for different functions
		self.func = "log"

	def forward_prop(self, train_in):
		"""
		Forward propogation to start making predictions
		INPUT: training input
		"""
		# move to hidden layer, multiplying the input by a weight and summed together
		# an additional bias term is added to explain neuron flexibility
		hidden_layer = np.dot(train_in, self.wh) + self.bh
		# output the non linear transformation of the neuron
		self.ah = self.function(hidden_layer)
		# multiply the hidden layer element by its weight and sum results together
		# add bias term
		out_layer = np.dot(self.ah, self.wo) + self.bo
		# output the non linear transformation of the neuron
		self.ao = self.function(out_layer)
		return self.ao

	def back_prop(self, train_in, train_out, lr):
		"""
		Optimize prediction by minimizing error with gradient descent
		INPUT: expected output and output activation
		"""
		# calculate the distance from the expected outcome and predicted values
		error = train_out - self.ao
		# compute gradient of the output layer
		slope_out = self.dx_function(self.ao)
		# compute gradient of the hidden layer
		slope_hidden = self.dx_function(self.ah)
		# compute weight modifications for output layer
		delta_out = error * slope_out
		# compute weight modification for hidden layer
		# error of hidden layer if back-propagated from error at output
		delta_hidden = np.dot(delta_out,self.wo.T) * slope_hidden
		# update weights dependending on learning rate and errors calculated
		self.wo += np.dot(self.ah.T,delta_out) * lr
		self.wh += np.dot(train_in.T, delta_hidden) * lr
		# update biases by summing up deltas
		self.bo += np.sum(delta_out, axis=0, keepdims=True) * lr
		self.bh += np.sum(delta_hidden, axis=0, keepdims=True) * lr
		return error

	def train(self,train_in, train_out, iters=5000, lr=0.5):
		"""
		Train neural network; go through N number of training cycles, 
		performing forward and backward propagation at each cycle,
		getting closer to learning the output at each iteration 
		
		INPUT: input training set, expected output, number of 
		iterations, and learning rate
		OUTPUT: trained model
		"""
		MSE = [float("inf")]*(iters)
		i = 0
		while i < iters:
			if i > 2 and (round(MSE[i-1],6) >= round(MSE[i-2],6) and MSE[i-1] < .00001):
				break
			self.forward_prop(train_in)
			error = self.back_prop(train_in, train_out,lr)
			MSE[i] = np.average(np.square(error))
			i+=1
		MSE = [x for x in MSE if x != float("inf")]
		return MSE

	def test(self, test_set):
		"""
		Test trained neural net on new input data.
		Use forward propagation to obtain the prediction
		for each test element. 
		INPUT: new, unseen testing data
		OUTPUT: predictions from the model
		"""
		predicted = np.array([])
		for i in range(0, test_set.shape[0]):
			predicted = np.append(predicted, self.forward_prop(test_set[i]))
		return predicted

	def function(self, x):
		"""
		Calculate activation function
		"""
		if self.func == "log":
			return logistic(x)
		elif self.func == "tanh":
			return tanh(x)
		elif self.func == "arctan":
			return arctan(x)
		return None

	def dx_function(self, x):
		"""
		Calculate activation function derivative
		"""
		if self.func == "log":
			return dx_logistic(x)
		elif self.func == "tanh":
			return dx_tanh(x)
		elif self.func == "arctan":
			return dx_arctan(x)
		return None



