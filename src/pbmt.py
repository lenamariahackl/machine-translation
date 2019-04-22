import numpy as np
import copy

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
                        if 0 < j_new < len(e2f):
                            if 0 < i_new < len(f2e):
                                if (not aligned_e(j_new, a) or not aligned_f(i_new, a)) and (j_new, i_new) in union_a:
                                    align(a, j_new, i_new)
    for j_new in range(0, len(e2f)):
        for i_new in range(0, len(f2e)):
            if (not aligned_e(j_new, a) or not aligned_f(i_new, a)) and (j_new, i_new) in union_a:
                align(a, j_new, i_new)
    return a


# Viterbi alignment
# find the most probable alignment for a given translation pair
# Output: dictionary viterbi_alignment src -> [tar]
def viterbi_alignment(src, tar, t, a):
    viterbi_alignment = {}
    for (j, src_j) in enumerate(src):
        max_value = -1
        max_index = None
        for (i, tar_i) in enumerate(tar):
            # TODO what if no alignment or (src_j, tar_i) not in t?
            if (src_j, tar_i) in t:
                value = t[(src_j, tar_i)] 
                 # TODO else what? should a be 0 then or be ignored?
                if (i, j, len(src), len(tar)) in a:
                    value *= a[(i, j, len(src), len(tar))]
                if max_value < value:
                    max_value = value
                    max_index = i
        viterbi_alignment[j] = max_index
    return viterbi_alignment


# Phrase extraction from symmetrized alignment table
# Input: word alignments a for sentence pairs (e,f)
# Output: set of phrase pairs BP
# S. 133 Figure 5.5
# TODO max phrase length
def extract(f, f_start, f_end, e, e_start, e_end, a):
	# check if at least one alignment point
	if f_end < 0: 
		return []
	# check if alignment points violate consistency
	for j in range(0, len(e)):
		for i in range(0, len(f)):
			if i in a.get(j,[]):
				if (f_start <= i and i <= f_end) and (j < e_start or j > e_end):                			
					return []
	# add phrase pairs (incl. additional unaligned f)
	E = []
	f_s = f_start
	first1 = True
	while not aligned_f(f_s, a) or first1:
		first1 = False
		f_e = f_end
		first2 = True
		while not aligned_f(f_e, a) or first2:
			first2 = False
			E.append((e[e_start:e_end + 1], f[f_s:f_e + 1]))
			f_e += 1
			if f_e >= len(f): break
		f_s -= 1
		if f_s < 0: break
	return E


def phrase_extraction(e_set, f_set, all_a):
	all_BP = []
	for index, a in all_a.items():
		e = e_set[index]
		f = f_set[index]
		BP = []
		for e_start in range(0, len(e)):
			for e_end in range(e_start, len(e)):
				# find the minimally matching foreign phrase
				(f_start, f_end) = (len(f) - 1, -1)
				for j in range(0, len(e)):
					for i in range(0, len(f)):
						if i in a.get(j,[]):
							if e_start <= j <= e_end:
								f_start = min(i, f_start)
								f_end = max(i, f_end)
				BP.append(extract(f, f_start, f_end, e, e_start, e_end, a))
		all_BP.append([item for sublist in BP for item in sublist])
	return [item for sublist in all_BP for item in sublist]


def count_phrases(all_BP):
    phrase_counts = {}
    for (a, b) in all_BP:
        key = str((a, b))
        phrase_counts[key] = phrase_counts.get(key, 0) + 1
    return phrase_counts


# Phrase translation probability
def PT_prob(e, f, phrase_counts):
	key = str((e, f))
	sum = 0
	for f_i in f:
		if str((e, [f_i])) in count:
			sum += phrase_counts[str((e, [f_i]))]
	if str((e, f)) in phrase_counts:
		return phrase_counts[key] / sum
	else:
		return 0


#bigram language model
def count_grams(data):
    unigram_counts = {}
    bigram_counts = {} 
    for sentence in data:
        unigram = sentence[len(sentence)-1]
        unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1
        for index in range(0, len(sentence)-1):
            bigram = (sentence[index], sentence[index+1])
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1
            unigram = sentence[index]
            unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1
    return unigram_counts, bigram_counts


#TODO is smoothing right?
# language model probability
def LM_prob(prev, cur, unigram_counts, bigram_counts):
    smoothed_bigram_counts = bigram_counts.get((prev, cur), 0) + 1     #smoothing
    smoothed_unigram_counts = unigram_counts.get(prev, 0) + len(unigram_counts)
    if smoothed_unigram_counts == 0:
        return 0
    else: 
        return float(smoothed_bigram_counts) / float(smoothed_unigram_counts)


#distance based reordering model d
#exponentially decaying cost function
def d(x):
    alpha = 0.5 # can be any value in [0,1]
    return alpha**abs(x)


def calc_vals_for_d(a, j, f):
    #pos of 1st word of f that translates to jth phrase in e
    i_first_occ = len(f)
    #TODO what to do if no first occurence at all
    for i in a.get(j,[]):
        if i <= i_first_occ: i_first_occ = i
    #pos of last word of f that translates to (jth - 1) phrase in e
    i_last_occ = -1
    if j > 0:
        for i in a.get(j-1,[]):
            if i >= i_last_occ: i_last_occ = i
    return i_first_occ, i_last_occ
