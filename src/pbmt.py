import numpy as np
import copy

# Phrase-based model

# a_hat = arg max_a p(e_j,f_i)
def align(e_set, f_set, t):
	all_aligned = {}
	for sentence in range(0, len(e_set)):
		e = e_set[sentence]
		f = f_set[sentence]
		aligned = np.zeros((len(e), len(f)))
		for i in range(0, len(f)):
			f_i = f[i]
			best_prob = 0
			best_j = 0
			for j in range(0, len(e)):
				e_j = e[j]
				if (e_j, f_i) in t and t[(e_j, f_i)] > best_prob:
					best_prob = t[(e_j, f_i)]
					best_j = j
			aligned[best_j][i] = 1
		all_aligned[sentence] = aligned
	return all_aligned

# Symmetrization of alignment
# for every sentence a alignment matrix esentence - fsentence
# 0 if not aligned, 1 if aligned
# 2 directions -> make intersection
# make union, neighbour logic
# for each sentence combine the two alignments using the word alignment algorithm
# >merged by taking the intersection or the union of alignment points of each alignment
# The algorithm starts with the intersection of the alignments. In
# the growing step, neighboring alignment points that are in the union but
# not in the intersection are added. In the final step, alignment points for
# words that are still unaligned are added.

def intersect(a, b):
	intersect = np.zeros(np.shape(a))
	for j in range(0, np.shape(a)[0]):
		for i in range(0, np.shape(a)[1]):
			intersect[j][i] = 1 if (a[j][i] == 1 and b[j][i] == 1) else 0
	return intersect


def union(a, b):
	union = np.zeros(np.shape(a))
	for j in range(0, np.shape(a)[0]):
		for i in range(0, np.shape(a)[1]):
			union[j][i] = 1 if (a[j][i] == 1 or b[j][i] == 1) else 0
	return union


def aligned_e(j, a):
	return np.sum(a[j]) != 0


def aligned_f(i, a):
	return np.sum([a[j][i] for j in range(0, np.shape(a)[0])]) != 0


def aligned(j, i, a):
	return a[j][i] == 1


def neighbour_point(j, i):
	neighbouring = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
	return [(a + j, b + i) for (a, b) in neighbouring]


def grow_diag_final(e2f, f2e):
	a = intersect(e2f, f2e)
	union_alignment = union(e2f, f2e)
	last_a = []
	# iterate until convergence
	while not np.array_equal(last_a, a):
		last_a = copy.deepcopy(a)
		for j in range(0, np.shape(e2f)[0]):
			for i in range(0, np.shape(e2f)[1]):
				if aligned(j, i, a):
					for (j_new, i_new) in neighbour_point(j, i):
						if j_new < len(a):
							if i_new < len(a[j_new]):
								if (not aligned_e(j_new, a) or not aligned_f(i_new, a)) and (j_new, i_new) in union_alignment:
									a[j_new][i_new] = 1
	for j_new in range(0, np.shape(a)[0]):
		for i_new in range(0, np.shape(a)[1]):
			if (not aligned_e(j_new, a) or not aligned_f(i_new, a)) and (j_new, i_new) in union_alignment:
				a[j_new][i_new] = 1
	return a


# Phrase extraction from symmetrized alignment table
# phrase extraction algorithm Use a suitable threshold for maximum phrase length and
# estimate the phrase translation probabilities.
# Phrase extraction algorithm:
# For each English phrase estart .. eend,
# the minimal phrase of aligned foreign words is identified fstart .. fend. Words in
# the foreign phrase are not allowed to be aligned with English words outside the
# English phrase. This pair of phrases is added, along with additional phrase pairs
# that include additional unaligned foreign words at the edge of the foreign phrase.
# Input: word alignment A for sentence pair (e,f)
# Output: set of phrase pairs BP

# TODO max phrase length
def extract(f, f_start, f_end, e, e_start, e_end, a):
	# check if at least one alignment point
	if f_end < 0:
		return []
	# check if alignment points violate consistency
	for j in range(0, np.shape(a)[0]):
		for i in range(0, np.shape(a)[1]):
			if a[j][i] == 1:
				if (f_start <= i <= f_end) and (j < e_start or j > e_end):
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
			if f_e >= np.shape(a)[1]: break
		f_s -= 1
		if f_s < 0: break
	return E


def phrase_extraction(e_set, f_set, all_alignments):
	all_BP = []
	for index, alignment in all_alignments.items():
		e = e_set[index]
		f = f_set[index]
		BP = []
		for e_start in range(0, len(e)):
			for e_end in range(e_start, len(e)):
				# find the minimally matching foreign phrase
				(f_start, f_end) = (len(f) - 1, -1)
				for j in range(0, np.shape(alignment)[0]):
					for i in range(0, np.shape(alignment)[1]):
						if alignment[j][i] == 1:
							if e_start <= j <= e_end:
								f_start = min(i, f_start)
								f_end = max(i, f_end)
				BP.append(extract(f, f_start, f_end, e, e_start, e_end, alignment))
		all_BP.append([item for sublist in BP for item in sublist])
	return all_BP


# Scoring Phrase Translations
# the translation probability φ(f|e) is estimated by the relative frequency
def φ(e, f, count):
	key = str((e, f))
	sum = 0
	for f_i in f:
		if str((e, [f_i])) in count:
			sum += count[str((e, [f_i]))]
	if str((e, f)) in count:
		return count[key] / sum
	else:
		return 0


#distance based reordering model d
#exponentially decaying cost function
def d(x):
    alpha = 0.5 # can be any value in [0,1]
    return alpha**abs(x)


def calc_vals_for_d(alignment, j, f):
    #pos of 1st word of f that translates to jth phrase in e
    i_first_occ = len(f)
    #TODO what to do if no first occurence at all
    for i, f_i in enumerate(alignment[j]):
        if f_i == 1:
            if i <= i_first_occ: i_first_occ = i
    #pos of last word of f that translates to (jth - 1) phrase in e
    i_last_occ = -1
    if j > 0:
        for i, f_i in enumerate(alignment[j-1]):
            if f_i == 1:
                if i >= i_last_occ: i_last_occ = i
    return i_first_occ, i_last_occ
