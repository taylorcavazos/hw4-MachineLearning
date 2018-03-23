# Test functions
from learn import nnet
from learn import handle_seq
from learn import nnet_helpers
import numpy as np
import matplotlib.pyplot as plt

def test_8x3x8_encoder():
	"""
	Test neural net implementation on the 8x3x8 encoder problem
	"""
	training = np.identity(8)
	autoencoder = nnet.NNET(8,3,8)
	mse = autoencoder.train(training, training, iters=5000, lr=2.5)
	nnet_helpers.plot_MSE(mse, "MSE_8x3x8_encoder.pdf", "Error Minimization 8x3x8 Encoder Problem")
	np.savetxt("8x3x8_output.txt", autoencoder.ao, fmt='%1.6f')
	assert np.array_equal(training, np.round(autoencoder.ao, 0))

def test_encode_DNA():
	"""
	Test encoding function to convert DNA sequence to binary output
	"""
	seq = "ACTGC"
	answer = np.array([0,0]+[0,1]+[1,1]+[1,0]+[0,1])
	result = handle_seq.encode_DNA(seq)
	assert np.array_equal(answer, result)