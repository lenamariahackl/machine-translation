import copy
import pickle
import os

DEBUG = False
MIN_PROB = 1e-12#float('-inf')

# amount the value is allowed to be off for convergence
# the smaller the longer the training takes
EPSILON = 1.0e-9

# test if the two sets are equivalent
def is_converged(t, last_t):
	for (e_j, f_i) in t.keys():
		if abs(t[(e_j, f_i)] - last_t[(e_j, f_i)]) > EPSILON: return False
	return True


# learning translation probabilitity distributions from sentence-aligned parallel text
# expectation maximization algorithm
# Input: set of sentence pairs (e,f), t_table to initialize t, number of maximum iterations
# optional: filenames to save t_table and a_table in files
# Output: translation prob. t (lexical translation) and a (alignment)
# S.99 Figure 4.7
# TODO e = [None] + e
def EM_IBM_Model_2(e_set, f_set, ibm1_t, max_steps, filename_t=None, filename_a=None):
	if DEBUG: print('start training IBM Model 2')
	# initialize t(e|f) with IBM Model 1
	t = copy.deepcopy(ibm1_t)
	# initialize a(i|j, l_e, l_f)
	e_lengths = [len(e) for e in e_set]
	f_lengths = [len(f) for f in f_set]
	a = {}
	for k in range(0, len(e_set)):  # for every sentence
		e = e_set[k]
		f = f_set[k]
		l_e = len(e)
		l_f = len(f)
		for j in range(0, l_e):
			for i in range(0, l_f):
				a[(i, j, l_e, l_f)] = 1.0 / (l_f + 1)
	# sets of distinct words
	last_t = {k:1 for k,_ in t.items()}  # to fail first comparison
	step = 0
	# iterate until convergence
	while not (is_converged(t, last_t)) and step < max_steps:
		# initialize
		count = {}
		total = {}
		total_a = {}
		count_a = {}
		for k in range(0, len(e_set)):
			e = e_set[k]
			f = f_set[k]
			l_e = len(e)
			l_f = len(f)
			# compute normalization
			s_total = {}
			for j in range(0, l_e):
				e_j = e[j]
				s_total[e_j] = 0
				for i in range(0, l_f):
					f_i = f[i]
					s_total[e_j] += t[(e_j, f_i)] * a[(i, j, l_e, l_f)]
			# collect counts
			for j in range(0, l_e):
				for i in range(0, l_f):
					e_j = e[j]
					f_i = f[i]
					c = t[(e_j, f_i)] * a[(i, j, l_e, l_f)] / s_total[e_j]
					count[(e_j, f_i)] = count.get((e_j, f_i), 0) + c
					total[f_i] = total.get(f_i, 0) + c
					count_a[(i, j, l_e, l_f)] = count_a.get((i, j, l_e, l_f), 0) + c
					total_a[(j, l_e, l_f)] = total_a.get((j, l_e, l_f), MIN_PROB) + c
		# estimate probabilities, max(t,MIN_PROB) cause they can never be 0
		last_t = copy.deepcopy(t)
		t = {x:0 for x in t}
		a = {x:0 for x in a}
		for (e_j, f_i) in count.keys():
			if count[(e_j, f_i)] != 0:
				t[(e_j, f_i)] = count[(e_j, f_i)] / total[f_i]
		for (i,j,l_e,l_f) in count_a.keys():
			if count_a[(i, j, l_e, l_f)] != 0:
				a[(i, j, l_e, l_f)] = count_a[(i, j, l_e, l_f)] / total_a[(j, l_e, l_f)]
		step += 1
		if DEBUG and step % 25 == 0: print('step', step,'of', max_steps)
	if DEBUG: print('IBM Model 2 training finished.')
	if filename_a is not None:
		if DEBUG: print('Save a table in', filename_a)
		path, _ = os.path.split(filename_a)
		os.makedirs(path, exist_ok=True)
		f2 = open(filename_a, "wb", pickle.HIGHEST_PROTOCOL)
		pickle.dump(a, f2)
		f2.close()
	if filename_t is not None:
		if DEBUG: print('Save t table in', filename_t)
		path, _ = os.path.split(filename_t)
		os.makedirs(path, exist_ok=True)
		f = open(filename_t, "wb", pickle.HIGHEST_PROTOCOL)
		pickle.dump(t, f)
		f.close()
	return t, a


# p(e|f) for IBM Model 2
# S.98 Figure 4.26 + Errata
#Input: sentences e and f, epsilon, t-table and a-table
def prob_e_given_f_2(e, f, epsilon, t, a):
	new = False
	all_new = True
	l_e = len(e)
	l_f = len(f)
	prod = 1
	for j in range(0, l_e):
		e_j = e[j]
		sum = 0.0
		for i in range(0, l_f):
			f_i = f[i]
			if (e_j, f_i) in t.keys() and (i, j, l_e, l_f) in a.keys():
				sum += t[(e_j, f_i)] * a[(i, j, l_e, l_f)]
		if sum != 0:
			prod *= sum # here underflow possible and a value <= 0
			all_new = False
		else:
			new = True
	if all_new:
		if DEBUG: print('None of the contained words were in the training set.')
	elif new:
		if DEBUG: print('Some contained words were not in the training set.')
	if prod == 1 and all_new: return 0.0
	else: return epsilon * prod
