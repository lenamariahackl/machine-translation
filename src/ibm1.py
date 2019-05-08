import copy

DEBUG = True

# test if the two sets are equivalent
def is_converged(t, last_t):
	return t == last_t


# learning translation probabilitity distributions from sentence-aligned parallel text
# expectation maximization algorithm
# Input: set of sentence pairs (e,f)
# Output: translation prob. t(e|f)
# S.91 Figure 4.3
def EM_IBM_Model_1(e_set, f_set, max_steps):
	if DEBUG: print('start training IBM Model 1')
	# sets of distinct words
	e_words = set([item for sublist in e_set for item in sublist])
	f_words = set([item for sublist in f_set for item in sublist])
	# initialize t(e|f) with uniform distribution
	t = {}  # {key: {key2: 1/len(e_words) for key2 in f_words} for key in e_words} #make code faster
	for k in range(0, len(e_set)):
		e = e_set[k]
		f = f_set[k]
		for e_j in e:
			for f_i in f:
				t[(e_j, f_i)] = 1 / len(e_words)  # TODO is this right? divide by 4
	last_t = {0: 1}  # to fail first comparison
	step = 0
	# iterate until convergence
	while not (is_converged(t, last_t)) and step < max_steps:
		# initialize
		count = {(e, f): 0 for f in f_words for e in e_words}
		total = {f: 0 for f in f_words}
		for k in range(0, len(e_set)):  # works cause number lines same
			e = e_set[k]
			f = f_set[k]
			# compute normalization
			s_total = {}
			for e_j in e:
				s_total[e_j] = 0
				for f_i in f:
					s_total[e_j] += t[(e_j, f_i)]
			# collect counts
			for f_i in f:
				for e_j in e:
					count[(e_j, f_i)] += t[(e_j, f_i)] / s_total[e_j]
					total[f_i] += t[(e_j, f_i)] / s_total[e_j]
		# estimate probabilities
		last_t = copy.deepcopy(t)
		for f_i in f_words:
			for e_j in e_words:
				if (e_j, f_i) in t:
					t[(e_j, f_i)] = count[(e_j, f_i)] / total[f_i]
		step += 1
		if DEBUG and step%25==0: print('step', step,'of', max_steps)
	if DEBUG: print('IBM Model 1 training finished.')
	return t


# p(e|f) for IBM Model 1
# S.90 Figure 4.10
def prob_e_given_f_1(e, f, epsilon, t):
	prod = 1
	for e_j in e:  # TODO starts at 1 not at 0!?
		sum = 0
		for f_i in f:
			if (e_j, f_i) in t:
				sum += t[(e_j, f_i)]
		prod *= sum
	return (epsilon / len(f) ** len(e)) * prod
# TODO without +1 for NONE right now
