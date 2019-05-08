import copy, ibm1
from ibm1 import is_converged

DEBUG = True
MIN_PROB = 1.0e-12
# learning translation probabilitity distributions from sentence-aligned parallel text
# expectation maximization algorithm
# Input: set of sentence pairs (e,f)
# Output: translation prob. t (lexical translation) and a (alignment)
# S.99 Figure 4.7
# TODO e = [None] + e
def EM_IBM_Model_2(e_set, f_set, t, max_steps):
	if DEBUG: print('start training IBM Model 2')
	# initialize t(e|f) with IBM Model 1
	t = copy.deepcopy(t)
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
	last_t = {}  # to fail first comparison
	step = 0
	# iterate until convergence
	while not (is_converged(t, last_t)) and step < max_steps:
		# initialize
		count = {}
		total = {}
		total_a = {}
		count_a = {}
		for l_e in e_lengths:
			for l_f in f_lengths:
				for j in range(0, l_e):
					total_a[(j, l_e, l_f)] = MIN_PROB
					for i in range(0, l_f):
						count_a[(i, j, l_e, l_f)] = 0
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
					count_a[(i, j, l_e, l_f)] += c
					total_a[(j, l_e, l_f)] += c
		# estimate probabilities, max(t,MIN_PROB) cause they can never be 0
		last_t = copy.deepcopy(t)
		for (e_j, f_i) in count.keys():
			t[(e_j, f_i)] = max(count[(e_j, f_i)] / total[f_i], MIN_PROB)
		for (i,j,l_e,l_f) in count_a.keys():
			a[(i, j, l_e, l_f)] = max(count_a[(i, j, l_e, l_f)] / total_a[(j, l_e, l_f)], MIN_PROB)
		step += 1
		if DEBUG and step % 25 == 0: print('step', step,'of', max_steps)
	if DEBUG: print('IBM Model 2 training finished.')
	return t, a


# p(e|f) for IBM Model 2
# S.98 Figure 4.26 + Errata
#Input: sentences e and f, epsilon, t-table and a-table
def prob_e_given_f_2(e, f, epsilon, t, a):
	l_e = len(e)
	l_f = len(f)
	prod = 1
	for j in range(0, l_e):  # TODO starts at 1 not at 0!
		e_j = e[j]
		sum = 0.0
		for i in range(0, l_f):
			f_i = f[i]
			if (e_j, f_i) in t.keys() and (i, j, l_e, l_f) in a.keys():
				sum += t[(e_j, f_i)] * a[(i, j, l_e, l_f)]
		prod *= sum
	return epsilon * prod
# TODO when removing += it's 1 and not 4 anymore which is good - so maybe dont sum over i?
