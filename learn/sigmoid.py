# File with sigmoid functions
import numpy as np
from scipy import integrate

def logistic(x):
	return 1/(1+np.exp(-x))

def dx_logistic(x):
	return x *(1-x)

def tanh(x):
	return np.tanh(x)

def dx_tanh(x):
	return 1 - (np.tanh(x)*np.tanh(x))

def arctan(x):
	return np.arctan(x)

def dx_arctan(x):
	return 1/(1+np.pow(x,2))