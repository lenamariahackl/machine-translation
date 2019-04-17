import copy, ibm1
from ibm1 import is_converged


# learning translation probabilitity distributions from sentence-aligned parallel text
# expectation maximization algorithm
# Input: set of sentence pairs (e,f)
# Output: translation prob. t (lexical translation) and a (alignment)
# S.99 Figure 4.7
# TODO e = [None] + e
def EM_IBM_Model_2(e_set, f_set, t, max_steps):
	# initialize t(e|f) with IBM Model 1
	t = t
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
				a[(i, j, l_e, l_f)] = 1 / (l_f + 1)
	# sets of distinct words
	e_words = list(set([item for sublist in e_set for item in sublist]))
	f_words = list(set([item for sublist in f_set for item in sublist]))
	last_t = {}  # to fail first comparison
	step = 0
	# iterate until convergence
	while not (is_converged(t, last_t)) and step < max_steps:
		# initialize
		count = {(e, f): 0 for f in f_words for e in e_words}
		total = {f: 0 for f in f_words}
		total_a = {}
		count_a = {}
		for l_e in e_lengths:
			for l_f in f_lengths:
				for j in range(0, l_e):
					total_a[(j, l_e, l_f)] = 0
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
					count[(e_j, f_i)] += c
					total[f_i] += c
					count_a[(i, j, l_e, l_f)] += c
					total_a[(j, l_e, l_f)] += c
		# estimate probabilities
		last_t = copy.deepcopy(t)
		for f_i in f_words:
			for e_j in e_words:
				if (e_j, f_i) in t:
					t[(e_j, f_i)] = count[(e_j, f_i)] / total[f_i]
		for l_e in e_lengths:
			for l_f in f_lengths:
				for i in range(0, l_f):
					for j in range(0, l_e):
						if (i, j, l_e, l_f) in a:
							a[(i, j, l_e, l_f)] = count_a[(i, j, l_e, l_f)] / total_a[(j, l_e, l_f)]
		step += 1
	return t, a


# returns most likely aligned input word position i for a given position j in output sentence
# TODO wrong, return max (a*t)!!!!
def a_fct(j, e, f, t):
	e_j = e[j]
	max_t = 0
	max_i = None
	for i in range(0, len(f)):
		f_i = f[i]
		if (e_j, f_i) in t and t[(e_j, f_i)] > max_t:
			max_t = t[(e_j, f_i)]
			max_i = i
	return max_i


# p(e|f) for IBM Model 2
# S.98 Figure 4.26
# TODO we can leave out for i in range, cause i isnt used?
# TODO but maybe we cant cause the sum changes eventhough!
def prob_e_given_f_2(e, f, epsilon, t, a):
	l_e = len(e)
	l_f = len(f)
	prod = 1
	for j in range(0, l_e):  # TODO starts at 1 not at 0!
		e_j = e[j]
		sum = 0
		for i in range(0, l_f):
			f_i = f[i]
			if (e_j, f_i) in t:
				f_aj = f[a_fct(j, e, f, t)]
				sum += t[(e_j, f_aj)] * a[(a_fct(j, e, f, t), j, l_e, l_f)]
		prod *= sum
	return epsilon * prod
# TODO when removing += it's 1 and not 4 anymore which is good - so maybe dont sum over i?
