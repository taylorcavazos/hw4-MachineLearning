# Run machine learning to predict transcription factor binding sites
import sys
from .handle_seq import *
from  .nnet import *
import pandas as pd
from .nnet_helpers import *

train_neg_file = sys.argv[1] # Negative training sequences
train_pos_file = sys.argv[2] # Positive training sequences
test_file = sys.argv[3] # Test sequences

######################## K-FOLDS CROSS VALIDATION FOR PARAMETER TUNING ##############################################
if len(sys.argv) > 4 and sys.argv[4] == "--optimize":
	# Loop through ratios of positive to negative sequences and perform cross validation to find 
	# best number of hidden layers and the best learning rate
	for r in [5,10]:
		# Read in sequences
		pos_seq, neg_seq, test_seq = read_seqs(train_pos_file, train_neg_file, test_file)
		# Filter negative sequenes to avoid class imbalance
		print("Number of RAP1 binding sequences: "+str(len(pos_seq)))
		print("Number of non-binding sequences: "+str(len(neg_seq)))
		neg_seq = filter_neg_seqs(pos_seq, neg_seq, ratio = r)
		print("Number of non-binding sequences after filtering and downsampling using ratio "+ str(r)+ ": "+str(len(neg_seq)))
		print("\n")

		# Cross validation
		cross_validate(pos_seq, neg_seq, "cross_val_summary_log_ratio"+str(r)+".txt")

else:
	# Read in sequences
	pos_seq, neg_seq, unknown_seq = read_seqs(train_pos_file, train_neg_file, test_file)
	# Select negative data randomly with ratio to positive seqs, select 17bp region,
	neg_seq = filter_neg_seqs(pos_seq, neg_seq, ratio = 5)
	# Shuffle and encode training and validation data
	train_seq, train_exp, test_seq, test_exp = preprocess(pos_seq, neg_seq, split=1)
	unknown_seq_encode = np.array([encode_DNA(seq) for seq in unknown_seq])
	# Use previously found validated parameters for training
	rap1_nnet = NNET(train_seq.shape[1],25,1)
	mse = rap1_nnet.train(train_seq, train_exp, iters=20000, lr=0.01)
	print(len(mse))
	plot_MSE(mse, "MSE_RAP1_binding.pdf", "Error Minimization in RAP1 Binding Site Problem")
	# Test data and output predictions
	out = open("/Users/student/Desktop/predictions.txt", "w")
	predictions = rap1_nnet.test(unknown_seq_encode)
	for p in range(0, len(predictions)):
		out.write(unknown_seq[p] + "\t" + str(predictions[p]))
		out.write("\n")
	out.close()