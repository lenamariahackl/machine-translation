import string
import numpy as np


def tokenize(lines, nr_used_sentences):
	lines = [line.rstrip() for line in lines[:nr_used_sentences]]  # remove \n
	words = [line.split() for line in lines]  # split by words
	unpunctuated = str.maketrans('', '', string.punctuation)  # remove punctuation & make Lowercase
	lower = [[word.translate(unpunctuated).lower() for word in sentence] for sentence in words]
	return lower


def read_in_corpus(nr_used_sentences, corpus_path):
	filename_e = "BU_en.txt"
	filename_f = "BU_tr.txt"

	e_file = open(corpus_path + filename_e, "r", encoding='utf-8-sig')
	e_lines = e_file.readlines()
	f_file = open(corpus_path + filename_f, "r", encoding='utf-8-sig')
	f_lines = f_file.readlines()

	f = tokenize(f_lines, nr_used_sentences)
	e = tokenize(e_lines, nr_used_sentences)
	return e, f


def split_dataset(dataset, training_split):
	#np.random.shuffle(dataset) #TODO less big vocabulary
	nr_train_samples = int(training_split * len(dataset))
	return dataset[:nr_train_samples], dataset[nr_train_samples:]
