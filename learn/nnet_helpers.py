from .nnet import *
from .handle_seq import *
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

def train_and_test(train_in, train_out, test_in,
		num_hidden, learn_rate, func="log", iters=10000):
	"""
	Train and test method simultaneously and output predictions
	INPUT: training data, expected outputs for training, and testing data
	number of hidden nodes and learning rate
	OUTPUT: predictions
	"""
	# Train model
	rap1_nnet = NNET(train_in.shape[1], num_hidden, 1)
	rap1_nnet.func = func
	rap1_nnet.train(train_in, train_out, iters, lr=learn_rate)

	# make predictions
	predictions = rap1_nnet.test(test_in)
	return predictions

def calc_static_sum_stats(expected, predicted):
	"""
	Count number of TP, TN, FP, and FN
	Calculate accuracy, TPR, and FPR
	"""
	TP, TN, FP, FN = 0, 0, 0, 0
	exp = np.reshape(expected, (len(expected),))
	for i in range(0, len(exp)):
		if exp[i] == 1 and exp[i] == np.round(predicted[i],0): TP +=1
		if exp[i] == 0 and exp[i] == np.round(predicted[i],0): TN +=1
		if exp[i] == 1 and exp[i] != np.round(predicted[i],0): FP +=1
		if exp[i] == 0 and exp[i] != np.round(predicted[i],0): FN +=1

	ACC = (TP+TN)/(TP+FP+FN+TN)

	if TP+FN != 0: TPR = TP/(TP+FN)
	else: TPR = 0

	if FP+TN != 0: FPR = FP/(FP+TN) 
	else: FPR = 0
	
	return ACC, str(TPR), str(FPR)

def k_folds(sequences,expected, lr, n_hidden,k=10):
	"""
	Perform cross validation using the k folds approach where data
	is split into training and testing sets k times using 100-k/k split
	"""
	accuracy, auc, tpr, fpr = [], [], [], []
	# Split data
	skf = StratifiedKFold(n_splits=k)
	for train_ind, test_ind in skf.split(sequences, expected):
		train_seq, test_seq = sequences[train_ind], sequences[test_ind]
		train_exp, test_exp = expected[train_ind], expected[test_ind]
		# train and test model
		predictions = train_and_test(train_seq, train_exp, test_seq, n_hidden, lr)
		# Caclulate true positive and false positive rate
		fpr_roc, tpr_roc, thresholds = metrics.roc_curve(test_exp, predictions, pos_label=1)
		auc.append(metrics.auc(fpr_roc, tpr_roc))
		ACC, TPR, FPR = calc_static_sum_stats(test_exp, predictions)
		# Add summary stats across splits
		accuracy.append(ACC)
		tpr.append(TPR) 
		fpr.append(FPR)

	return np.average(accuracy), np.average(auc), tpr, fpr 

def cross_validate(pos_seqs, neg_seqs, outfile, iters=10000):
	"""
	Perform cross validation for a range of learning rates and hidden layers.
	Use kfold cross validation and output average accuracy
	"""
	train_seq, train_exp, test_seq, test_exp = preprocess(pos_seqs, neg_seqs, split=1)
	with open(outfile, "w") as results:
		results.write("hidden_layers\tlearn_rate\tavg_accuracy\tavg_AUC\tTPR\tFPR\n")
		for num_hidden in range(1,30, 4):
			print(num_hidden)
			for learn_rate in np.arange(0.01, 0.5, 0.05):
				print(learn_rate)
				avg_accuracy, avg_auc, tprs, fprs = k_folds(train_seq, train_exp, learn_rate, num_hidden)
				results.write("\t".join([str(num_hidden), str(learn_rate), str(avg_accuracy),
					str(avg_auc), " ".join(tprs), " ".join(fprs)]))
				results.write("\n")

def plot_roc(predictions, exp_test, outfile):
	"""
	plot ROC curve for predictions and expected values to asses model performance
	"""
	fpr, tpr, thresholds = metrics.roc_curve(exp_test, predictions, pos_label=1)
	AUC = metrics.auc(fpr, tpr)
	plt.plot(fpr, tpr)
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("ROC Training Data (80/20) Split with AUC: " + str(AUC))
	plt.savefig(outfile, format="pdf")
	plt.close()
	return

def plot_MSE(mse, outfile, title):
	"""
	plot mean squared error for training iterations to determine model convergence
	"""
	plt.figure(figsize=(10,5))
	plt.plot(np.arange(len(mse)), mse)
	plt.xlabel("Iterations", fontsize=12)
	plt.ylabel("Mean Squared Error", fontsize=12)
	plt.title(title, fontsize=14)
	plt.savefig(outfile, type = "pdf")
	plt.close()




