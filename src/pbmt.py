import numpy as np
import copy

MIN_PROB = 1.0e-12
DEBUG = True
# Phrase-based model

def intersect(e2f, f2e):
	intersection = {}
	for j in range(len(e2f)):
		i = e2f[j]
		if i != None and j == f2e[i]:
			align(intersection, j, i)
	return intersection


def union(e2f, f2e):
	union = {}
	for j in range(len(e2f)):
		i = e2f[j]
		if i != None:
			align(union, j, i)
	for i in range(len(f2e)):
		j = f2e[i]
		if j != None:
			align(union, j, i)
	return union


def aligned_e(j, a):
	return a.get(j,None) != None


def aligned_f(i, a):
	for j in range(0, len(a)):
		a[j] = a.get(j,[])
		if i in a[j]: return True
	return False


def aligned(j, i, a):
	a[j] = a.get(j,[])
	return i in a[j]


def neighbour_point(j, i):
	neighbouring = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
	return [(a + j, b + i) for (a, b) in neighbouring]


def align(a, j, i):
	a[j] = a.get(j,[])
	if i not in a[j]: a[j].append(i)


# Symmetrization of alignment using the word alignment algorithm
# for every sentence return alignment dict e -> f
# S. 118 Figure 4.14 GROW-DIAG-FINAL(e2f,f2e)
def combine(e2f, f2e):
	intersect_a = intersect(e2f, f2e)
	union_a = union(e2f, f2e)
	a = intersect_a
	last_a = {}
	# iterate until convergence
	while not last_a == a:
		last_a = copy.deepcopy(a)
		for j in range(0, len(e2f)):
			for i in range(0, len(f2e)):
				if aligned(j, i, a):
					for (j_new, i_new) in neighbour_point(j, i):
						if 0 <= j_new < len(e2f):
							if 0 <= i_new < len(f2e):
								if (not aligned_e(j_new, a) or not aligned_f(i_new, a)) and i_new in union_a.get(j_new,[]):
									align(a, j_new, i_new)
	for j_new in range(0, len(e2f)):
		for i_new in range(0, len(f2e)):
			if (not aligned_e(j_new, a) or not aligned_f(i_new, a)) and i_new in union_a.get(j_new,[]):
				align(a, j_new, i_new)
	return a


# Viterbi alignment
# find the most probable alignment for a given translation pair out of training set
# Output: dictionary viterbi_alignment src -> [tar]
def viterbi_alignment(src, tar, t, a):
	viterbi_alignment = {}
	for (j, src_j) in enumerate(src):
		max_value = -1
		max_index = None
		for (i, tar_i) in enumerate(tar):
			# (src_j, tar_i) is in t always cause we only call this method with the training data
			if (src_j, tar_i) in t:
				value = t[(src_j, tar_i)]
				if (i, j, len(src), len(tar)) in a:
					value *= a[(i, j, len(src), len(tar))]
				if value >= max_value:
					max_value = value
					max_index = i
		viterbi_alignment[j] = max_index
	return viterbi_alignment


# Phrase extraction from symmetrized alignment table
# extracts phrases and at the same time also counts them
# max_phrase_length is for the English phrase, if the foreign equivalent is longer that's fine
# Input: word alignments a for sentence pairs (e,f)
# Output: set of phrase pairs BP
# S. 133 Figure 5.5
def phrase_extraction(e_set, f_set, all_a, max_phrase_length):
	phrase_counts = {}
	for index, a in all_a.items():
		e = e_set[index]
		f = f_set[index]
		for e_start in range(0, len(e)):
			for e_end in range(e_start, min(len(e), e_start + max_phrase_length)):
				# find the minimally matching foreign phrase
				(f_start, f_end) = (len(f) - 1, -1)
				for j in range(0, len(e)):
					for i in range(0, len(f)):
						if i in a.get(j,[]):
							if e_start <= j and j <= e_end:
								f_start = min(i, f_start)
								f_end = max(i, f_end)
				consistent = True
				# check if alignment points violate consistency
				for j in range(0, len(e)):
					for i in range(0, len(f)):
						if i in a.get(j,[]):
							if (f_start <= i and i <= f_end) and (j < e_start or j > e_end):
								consistent = False
								# check if at least one alignment point
				if f_end >= 0 and consistent:
					# add phrase pairs (incl. additional unaligned f)
					f_s = f_start
					first1 = True
					while not aligned_f(f_s, a) or first1:
						first1 = False
						f_e = f_end
						first2 = True
						while not aligned_f(f_e, a) or first2:
							first2 = False
							index_e = str(e[e_start:e_end + 1])
							index_f = str(f[f_s:f_e + 1])
							phrase_counts[index_e] = phrase_counts.get(index_e, {})
							phrase_counts[index_e][index_f] = phrase_counts[index_e].get(index_f, 0) + 1
							f_e += 1
							if f_e >= len(f): break
						f_s -= 1
						if f_s < 0: break
	return phrase_counts


# Phrase translation probability
# given the counted phrase pairs the phrase translation probability distribution is estimated by relative frequency
# reflects the probability that phrases e and f are translation equivalents
# For each sentence pair, we extract a number of phrase pairs.
# Then, we count in how many sentence pairs a particular phrase pair is
# extracted and store this number in count(e,f).
def PT_prob(e, f, phrase_counts):
	if str(e) in phrase_counts and str(f) in phrase_counts[str(e)]:
		sum = 0
		for key, count in phrase_counts[str(e)].items():
			sum += count
		if DEBUG: print('calculate PT_prob for',e,f,'=', phrase_counts[str(e)][str(f)] / sum)
		return phrase_counts[str(e)][str(f)] / sum
	else:
		if DEBUG: print('calculate PT_prob for',e,f,'=',MIN_PROB)
		print('tutu',str(e) in phrase_counts, str(e), str(f))
		if str(e) in phrase_counts: print(phrase_counts[str(e)])
		return MIN_PROB
#TODO what if phrase not in phrasetable


# prepare data by adding [START] and [END] markers before/after every sentence
def add_markers(e):
	e_m = [['START'] + e_j + ['END'] for e_j in e]
	return e_m


# count bigrams and unigrams in data
def count_grams(data):
	if DEBUG: print('count_grams')
	data_m = add_markers(data)
	unigram_counts = {}
	bigram_counts = {}
	for sentence in data_m:
		unigram = sentence[len(sentence)-1]
		unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1
		for index in range(0, len(sentence)-1):
			bigram = (sentence[index], sentence[index+1])
			bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
			unigram = sentence[index]
			unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1
	return unigram_counts, bigram_counts


# bigram language model with Add One smoothing
# calculates LM probability for 2 words given unigram & bigram counts
def LM_prob(prev, cur, unigram_counts, bigram_counts):
	LM_prob_ = 1.0
	prev_unigram_count = unigram_counts.get(prev, 0)
	bigram_count = bigram_counts.get((prev, cur), 0)
	if bigram_count > 0:
		LM_prob_ *= bigram_count + 1
	LM_prob_ /= (prev_unigram_count + len(unigram_counts))
	if DEBUG: print('LM_prob(', cur, '|', prev, ')=', LM_prob_)
	return LM_prob_


# distance based reordering model d
# exponentially decaying cost function
# reflects the probability that a phrase at position i is translated into a phrase at position j
# called with (pos of 1st word of f that translates to jth phrase in e) - (pos of last word of f that translates to (jth - 1) phrase in e) - 1
def d(x):
	alpha = 0.5 # can be any value in [0,1]
	if DEBUG: print('d(',x,')=',alpha**abs(x))
	return alpha**abs(x)

# format aligned_phrases {e_to: (e_from, f_from, f_to)}
def prob_e_given_f(e, f, phrase_counts, unigram_counts, bigram_counts, aligned_phrases):
	prob_f_given_e = 1
	for e_to, (e_from, f_from, f_to) in aligned_phrases.items(): # for every phrase (as chosen manually)
		e_j = e[e_from:e_to+1]
		f_to_min_1 = -1
		if e_from > 0:
			(_,_, f_to_min_1) = aligned_phrases[e_from-1]
		f_i = f[f_from:f_to+1]
		prob_f_given_e *= PT_prob(e_j, f_i, phrase_counts)
		prob_f_given_e *= d(f_from - f_to_min_1 - 1)
	P_LM = 1
	for j, e_j in enumerate(e+['END']):
		if j > 0: prev = e[j-1]
		else: prev = 'START'
		P_LM *= LM_prob(prev, e_j, unigram_counts, bigram_counts)
	prob = prob_f_given_e * P_LM
	return prob