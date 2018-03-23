# File to read in input sequences for transcription factor binding sites
from Bio import SeqIO
from Bio.Seq import Seq
import re
import numpy as np
from scipy.spatial.distance import hamming
import pandas as pd

def read_seqs(pos_seq_file, neg_seq_file, test_file):
	"""
	Read in training and testing sequences
	INPUT: positive train file, negative train file, and test file
	OUTPUT: sequences
	"""
	pos_seqs = open(pos_seq_file).read().splitlines()
	test_seqs = open(test_file).read().splitlines()
	if ".fa" in neg_seq_file:
		neg_seqs = [str(fasta.seq) for fasta in SeqIO.parse(open(neg_seq_file), "fasta")]
	elif ".txt" in neg_seq_file:
		neg_seqs = open(neg_seq_file).read().splitlines()
	
	return pos_seqs, neg_seqs, test_seqs

def filter_neg_seqs(pos_seqs, neg_seqs, bp=17, ratio = 4):
	"""
	Drop negative sequences that are a sub-sequence of the positive sequences
	INPUT: positive sequences and negative sequences, length of bases to keep, and ratio for
	number of negative sequences to positive sequences
	OUTPUT: negative sequences that are not similar to positive sequences and 
	are the same number of base pairs as positive sequences
	"""
	# Drop negative sequences that match positive sequence completely
	for pos in pos_seqs:
		for neg in neg_seqs:
			if re.search(pos, neg):
				neg_seqs.remove(neg)
	# downsampling of negative sequences
	neg_keep = int(len(pos_seqs)*ratio)
	neg_seqs_sub = np.random.choice(neg_seqs, size=neg_keep, replace=False)
	# neg_keep = pd.DataFrame(AC_content(neg_seqs, "/Users/student/Desktop/neg_AC.txt"), index = neg_seqs)
	# pos_out = AC_content(pos_seqs, "/Users/student/Desktop/pos_AC.txt")
	# neg_seqs_sub = list(neg_keep[neg_keep.iloc[:,0] < 0.53].index)

	short_neg = []
	for neg in neg_seqs_sub:
		rand_start = np.random.randint(0, len(neg)-bp+1)
		short_neg.append(neg[rand_start:rand_start+bp])
	return short_neg


def encode_DNA(seq):
	"""
	Convert DNA sequence to binary values for input into neural net
	INPUT: DNA sequence
	OUTPUT: binary encoding of sequence
	"""
	seq2bin_dict = {'A':[0,0], 'C':[0,1], 'G':[1,0], 'T':[1,1]}
	return np.array(sum([seq2bin_dict.get(nuc) for nuc in seq], []))

def split_data(pos_seqs, neg_seqs, split):
	"""
	Split data with known outcomes into training and testing data
	INPUT: positive examples, negative examples, and percent to keep for training
	OUTPUT: positive and negative training sets and
	positive and negative testing sets
	"""
	pos_size, neg_size = int(len(pos_seqs)*split), int(len(neg_seqs)*split)
	train_pos = np.random.choice(pos_seqs, size=pos_size, replace=False)
	train_neg = np.random.choice(neg_seqs, size=neg_size, replace=False)
	test_pos = list(set(pos_seqs)-set(train_pos))
	test_neg = list(set(neg_seqs)-set(train_neg))
	return train_pos, train_neg, test_pos, test_neg

def combine_and_shuffle(pos, neg):
	combined = np.concatenate((pos, neg))
	expected = np.append(np.array([[1]]*len(pos)), np.array([[0]]*len(neg)))
	shuf_combined, shuf_expected = shuffle(combined, expected)
	return shuf_combined, np.reshape(shuf_expected, (len(shuf_expected),1))

def preprocess(pos_seqs, neg_seqs, split=0.8):
	"""
	process, split, and encode data
	INPUT: positive and negative sequences
	OUTPUT: training and testing encoded sequences
	"""
	# split the data into training and testing sets
	train_p_l, train_n_l, test_p_l, test_n_l = split_data(pos_seqs, neg_seqs, split)
	train_p_b = np.array([encode_DNA(seq) for seq in train_p_l])
	train_n_b = np.array([encode_DNA(seq) for seq in train_n_l])
	
	# encode dna from nucleotides to binary output
	test_p_b = np.array([encode_DNA(seq) for seq in test_p_l])
	test_n_b = np.array([encode_DNA(seq) for seq in test_n_l])

	# Combine training positive and negative sequences
	train_seq, train_exp = combine_and_shuffle(train_p_b, train_n_b)
	if split != 1: 
		test_seq, test_exp = combine_and_shuffle(test_p_b, test_n_b)
	else:
		test_seq, test_exp = [],[] 

	return train_seq, train_exp, test_seq, test_exp

def shuffle(sequences, phenotype):
	"""
	Shuffle inputs and outputs
	"""

	seqs = pd.DataFrame(sequences)
	phen = pd.DataFrame(phenotype)
	df = pd.concat([seqs,phen], axis=1)
	df_shuf = df.sample(frac=1)
	shuf_seqs = np.array(df_shuf.iloc[:,:-1])
	shuf_phen = np.array(df_shuf.iloc[:,-1])
	return shuf_seqs, shuf_phen