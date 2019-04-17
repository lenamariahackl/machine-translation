import ibm2, pbmt, utils, test
import numpy as np
from ibm2 import *
from pbmt import *
from utils import *
from test import *

nr_used_sentences = 30  # of 688670 lines
path = "./Paralel Corpus/"
e, f = read_in_corpus(nr_used_sentences, path)

training_split = 0.9
e_train, e_test = split_dataset(e, training_split)
f_train, f_test = split_dataset(f, training_split)

e_train, f_train = initialize_test_set()  # to test

max_steps = 50

e_words = set([item for sublist in e_train for item in sublist])
f_words = set([item for sublist in f_train for item in sublist])

print(e_train)
t = EM_IBM_Model_1(e_train, f_train, max_steps)

compare_ibm_1(t, max_steps, e_train, f_train)  # to test

# TODO my a is 1 when it should be low :(
t, a = EM_IBM_Model_2(e_train, f_train, max_steps)
compare_ibm_2(t, max_steps, a, e_train, f_train)  # to test
print('------')
t, a = EM_IBM_Model_2(f_train, e_train, max_steps)
compare_ibm_2(t, max_steps, a, f_train, e_train)

# Testing
for i in range(0, len(f_test)):
	f = f_test[i]
	e = e_test[i]
# arg max_e p(e)p(f|e) decoding manually #TODO how???
# calculate p(e|f)
# print(e, prob_e_given_f(e, f, 0.001, t))

f = ['das', 'Buch']
e = ['the', 'book']
print('p(', e, '|', f, ') =', prob_e_given_f_1(e, f, 1, t))
print('a: p(', e, '|', f, ') =', prob_e_given_f_2(e, f, 1, t, a))
e = ['kgefw', 'fek']
print('p(', e, '|', f, ') =', prob_e_given_f_1(e, f, 1, t))
print('a: p(', e, '|', f, ') =', prob_e_given_f_2(e, f, 1, t, a))

###############################################

# Phrase - based MT

# IBM Models for word alignment
t_e2f = EM_IBM_Model_1(e_train, f_train, max_steps)
all_aligned_e2f = align(e_train, f_train, t_e2f)
t_f2e = EM_IBM_Model_1(f_train, e_train, max_steps)
all_aligned_f2e = align(f_train, e_train, t_f2e)

# transpose so that alignments are same shape
all_aligned_f2e = [np.transpose(f2e) for f2e in list(all_aligned_f2e.values())]
for sentence in range(0, len(e_train)):
	print(e_train[sentence], f_train[sentence])
	print(all_aligned_e2f[sentence])
	print(all_aligned_f2e[sentence])
# x foreign, y English

# call for every sentence pair
all_alignments = {}
for index in range(0, len(e_train)):
	e2f = all_aligned_e2f[index]
	f2e = all_aligned_f2e[index]
	alignment = grow_diag_final(e2f, f2e)
	print(alignment)
	all_alignments[index] = alignment

e_train = [['house', 'the', 'the'], ['the', 'the', 'book'], ['a', 'book']]
f_train = [['das', 'Haus'], ['das', 'Buch'], ['ein', 'Buch']]
"""""
e_train = [['michael','assumes','that','he','will','stay','in','the','house']]
f_train = [['michael','geht','davon','aus',',','dass','er','im','haus','bleibt']]
all_alignments = {0:[[1,0,0,0,0,0,0,0,0,0],
                   [0,1,1,1,0,0,0,0,0,0],
                   [0,0,0,0,0,1,0,0,0,0],
                   [0,0,0,0,0,0,1,0,0,0],
                   [0,0,0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,0,0,0,0,1],
                   [0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,1,0,0],
                   [0,0,0,0,0,0,0,0,1,0]]}
"""""

all_BP = phrase_extraction(e_train, f_train, all_alignments)
# for BP in all_BP:
print(all_BP)

# probabilistic phrase translation table
# For each sentence pair, we extract a number of phrase pairs.
# Then, we count in how many sentence pairs a particular phrase pair is
# extracted and store this number in count(e,f).
all_BP_flat = [item for sublist in all_BP for item in sublist]
count = {}
for (a, b) in all_BP_flat:
	# key = (frozenset(a),frozenset(b))
	key = str((a, b))
	count[key] = count.get(key, 0) + 1
print(count)

print(φ(['house'], ['Haus'], count))


# print(φ(['the','house'],['das','Haus']))
# TODO translation probability is not working for phrases containing several words
# maybe other data structure ['the','house']:[['das','Haus'],['ein','Haus']]
# faster sometimes but slower if translation in other direction

# TODO For the estimation of the phrase translation probabilities, not all
# phrase pairs have to be loaded into memory. It is possible to efficiently
# estimate the probability distribution by storing and sorting the extracted
# phrases on disk. Similarly, when using the translation table for the translation of a single sentence, only a small fraction of it is needed and may
# be loaded on demand.
# think about saving stuff in file
# • Phrase translation table typically bigger than corpus
# ... even with limits on phrase lengths (e.g., max 7 words)
# → Too big to store in memory?
# • Solution for training
# – extract to disk, sort, construct for one source phrase at a time
# • Solutions for decoding
# – on-disk data structures with index for quick look-ups
# – suffix arrays to create phrase pairs on demand
# TODO use numpy for everything


# phrase-based statistical machine translation model
# • the phrase translation table φ(f |e);
# • the reordering model d;
# • the language model pLM(e).
# e_best = argmax_e prod_i φ(f|e) * d(start_i - end_i_min_1 - 1) * prod_i PLM(e_i|e_i_min_1)

# distance based reordering model d
# This model gives a cost linear to the reordering distance. For instance, skipping over two words costs twice as much as skipping over one word.
# <source-phrase> & <target-phrase> : <weight-1> <weight-2> ... <weight-k> for k of 6 / 8
def d(x):
	return 0  # alpha**|x| #TODO i dont get it

# bigram language model #TODO i dont get it
# <probability> <word-1><word-2>  <back-off-weight>
# eg -0.2359 When will 0.1011
# test the model using the test data.
# for a source sentence, you can generate some example target sentences (word sequences)
# and phrases “manually”; i.e. you will do the decoding process manually.
# That is, for a source sentence, generate a possible target sentence,
# divide both sentences into phrases as you wish, and assume that you know the phrase
# correspondences. Then, calculate the probability of the target sentence given the source
# sentence using the p(f|e)*p LM (e) equation (i.e. using the standard model with three
# components).

# You can train a simple bigram language model and a reordering model.

# Translation model - provides phrases in the source language with learned possible target language
# translations and the probabilities thereof.
# Reordering model - stores information about probable translation orders of the phrases within the
# source text, based on the observed source and target phrases and their alignments.
# Language model - reflects the likelihood of this or that phrase in the target language to occur.
# In other words, it is used to evaluate the obtained translation for being "sound" in the target language.
# Note that, the language model is typically learned from a different corpus in a target language.

# With these three models at hand one can perform decoding, which is a synonym to a translation process.
# SMT decoding is performed by exploring the state space of all possible translations and reordering of the
# source language phrases within one sentence. The purpose of decoding, as indicated by the maximization
# procedure at the bottom of the figure above, is to find a translation with the largest possible probability.

# QUESTIONS

# IBM Model 2: Do I calculate a correctly?
# > compare with other implementations
# > NOOOOO a is wrong :/

# p(e|f) for IBM Model 2: Gives result 4 should be 1?
# > compare with other implementations

# TODO did I have to implement all that?
# what exactly do i have to do? first train, then do what with t?
# decoding = calculating max p(e|f)?

# how to output the Viterbi alignment of the last iteration IBM 2?

# how to implement distance based reordering model?

# how to implement bigram language model?
