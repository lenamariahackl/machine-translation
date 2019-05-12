import nltk
import numpy as np
import math
from nltk.translate import AlignedSent, IBMModel1, IBMModel2

DEBUG = False


def initialize_test_sets():
	e_train = [['house','the','the'],['the','the','book'],['a','book'],['house']]
	f_train = [['das','Haus'],['das','Buch'],['ein','Buch'],['Haus']]
	all_alignments_train = { 0: {0:[1,2], 1:[0]}, 1: {0:[0,1], 1:[2]}, 2: {0:[0], 1:[1]}, 3:{0:[0]} }
	e_test = [['the'],['the','the','house','a','book'],['book','a'],['book','a'],['book','the'],['kgefw', 'fek']]
	f_test = [['das'],['das','ein','Buch','Haus'],['Buch','ein'],['ein','Buch'],['Buch','ein'],['Buch','ein']]
	all_alignments_test = { 0: {0:[0]}, 1: {0:[0,1], 1:[3], 2:[4], 3:[2]}, 2: {0:[0],1:[1]}, 3: {0:[1],1:[0]}, 4: {0:[0]}, 5: {} } #TODO what do I want, should everything be aligned even if just random? {0:[0],1:[0]}, 5: {0:[0],1:[0]}
	return e_train, f_train, all_alignments_train, e_test, f_test, all_alignments_test


def dicts_for_train_comparison(e_train, f_train):
	en_word_dict = {}
	en_lang_order = 0
	tk_word_dict = {}
	tk_lang_order = 0
	for sentence in e_train:
		for token in sentence:
			if token not in en_word_dict:
				en_word_dict[token] = en_lang_order
				en_lang_order += 1
	for sentence in f_train:
		for token in sentence:
			if token not in tk_word_dict:
				tk_word_dict[token] = tk_lang_order
				tk_lang_order += 1
	return en_word_dict, tk_word_dict


def test_sets_to_aligned(src, tar):
	bitext = []
	for sentence in range (0, len(src)):
		bitext.append(AlignedSent(tar[sentence], src[sentence]))
	return bitext


def compare_t_table(ibm, t):
	correct = True
	correct_t = 0
	for (e_j,f_i) in t:
		bool_ = (ibm.translation_table[f_i][e_j] > 0.7) == (t[(e_j,f_i)] > 0.7)
		if bool_ == False:
			if DEBUG: print('wrong t:', t[(e_j,f_i)], '!=',ibm.translation_table[f_i][e_j],'for',e_j, f_i,)
			correct = False
		else:
			correct_t += 1
	if correct: print('All t values were correct.\n')
	else: print('t values ', 100 * (correct_t / len(t)), '% correct,',correct_t,'values wrong.\n')


#compare t table to IBM Model 1 implementation of nltk library
def compare_ibm_1_nltk(t, max_steps, src, tar):
	print('Compare my IBM Model 1 to nltk library:')
	aligned = test_sets_to_aligned(src, tar)
	ibm1 = IBMModel1(aligned, max_steps)
	compare_t_table(ibm1, t)


#compare t and a table to IBM Model 2 implementation of nltk library
def compare_ibm_2_nltk(t, max_steps, a, src, tar):
	print('Compare my IBM Model 2 to nltk library:')
	aligned = test_sets_to_aligned(src, tar)
	ibm2 = IBMModel2(aligned, max_steps)
	compare_t_table(ibm2, t)
	correct = True
	correct_a = 0
	for (i,j,l_e,l_f) in a:
		bool_ = (a[(i,j,l_e,l_f)] > 0.7) == (ibm2.alignment_table[j][i][l_f][l_e] > 0.7)
		if bool_ == False:
			if DEBUG: print('wrong a:', a[(i,j,l_e,l_f)], '!=',ibm2.alignment_table[j][i][l_f][l_e],'for i',i,' j',j,' l_e',l_e,' l_f',l_f)
			correct = False
		else:
			correct_a += 1
			#print(' a:', a[(i,j,l_e,l_f)], 'for i',i,' j',j,' l_e',l_e,' l_f',l_f)
	if correct: print('All a values were correct.\n')
	else: print('a values ', 100 * (correct_a / len(a)), '% correct,',correct_a,'values wrong.\n')


def compare_a_ibm_2_train(t, max_steps, a, src, tar):
	print('Compare my IBM Model 2 to train() implementation:')
	# for (i,j,l_e,l_f), val in a.items():
	#    if val != 0: print('my',i,j,l_e,l_f, val)
	max_le = max([len(e) for e in src])
	max_lf = max([len(f) for f in tar])
	en_word_dict, tk_word_dict = dicts_for_train_comparison(src, tar)
	num_of_e_word = len(en_word_dict)
	num_of_f_word = len(tk_word_dict)
	t_e2f_ibm1_matrix = np.full((num_of_e_word, num_of_f_word), 0, dtype=float)
	for (e_j, f_i), t_val in t.items():
		t_e2f_ibm1_matrix[en_word_dict[e_j]][tk_word_dict[f_i]] = t_val
	t_e_f_mat, a_i_le_lf_mat = train(t_e2f_ibm1_matrix, en_word_dict, tk_word_dict, src, tar, max_le, max_lf,
	                                 max_steps/6)
	#less steps (divide by 6) for other implementation as it's so fuckin slow
	correct0 = 0
	sum = 0
	for i, REST1 in enumerate(a_i_le_lf_mat):
		for j, REST2 in enumerate(REST1):
			for l_f, REST3 in enumerate(REST2):
				for l_e, val in enumerate(REST3):
					#if val != 0: print('ot',i,j,l_e+1,l_f+1, val)
					bool_ = (a.get((i, j, l_e + 1, l_f + 1), 0) > 0.7) == (val > 0.7)
					if bool_ == False:
						if DEBUG: print('wrong a:', a.get((i, j, l_e + 1, l_f + 1), 0), '!=', val, 'for i', i, ' j', j,
						      ' l_e', l_e, ' l_f', l_f)
					else:
						correct0 += 1
					sum += 1
					# print('both a:', val, i, j, l_e, l_f)#'for i',i,' j',j,' l_e',l_e,' l_f',l_f)
	#nr_el_a_mat = len(a_i_le_lf_mat) * len(a_i_le_lf_mat[0]) * len(a_i_le_lf_mat[0][0]) * len(a_i_le_lf_mat[0][0][0])
	print('a values ', 100 * (correct0 / sum), '% correct,',correct0,'values wrong.\n')


def compare_a_nltk_train(t_ibm1, max_steps, src, tar):
	print('Compare nltk to train() implementation:')
	#train() implementation
	max_le = max([len(e) for e in src])
	max_lf = max([len(f) for f in tar])
	en_word_dict, tk_word_dict = dicts_for_train_comparison(src, tar)
	num_of_e_word = len(en_word_dict)
	num_of_f_word = len(tk_word_dict)
	t_e2f_ibm1_matrix = np.full((num_of_e_word, num_of_f_word), 0, dtype=float)
	for (e_j, f_i), t_val in t_ibm1.items():
		t_e2f_ibm1_matrix[en_word_dict[e_j]][tk_word_dict[f_i]] = t_val
	t_e_f_mat, a_i_le_lf_mat = train(t_e2f_ibm1_matrix, en_word_dict, tk_word_dict, src, tar, max_le, max_lf,
	                                 max_steps / 6)
	#nltk implementation
	aligned = test_sets_to_aligned(src, tar)
	ibm2 = IBMModel2(aligned, max_steps)
	a = ibm2.alignment_table
	t = ibm2.translation_table
	correct0 = 0
	sum = 0
	for i, REST1 in enumerate(a_i_le_lf_mat):
		for j, REST2 in enumerate(REST1):
			for l_f, REST3 in enumerate(REST2):
				for l_e, val in enumerate(REST3):
					#if val != 0:   print('ot',i,j,l_e+1,l_f+1, val)
					bool_ = (a[j][i][l_f+1][l_e+1] > 0.7) == (val > 0.7)
					if bool_ == False:
						if DEBUG: print('wrong a:', a[j][i][l_f+1][l_e+1], '!=', val, 'for i', i, ' j',j, ' l_e', l_e, ' l_f', l_f)
					else:
						correct0 += 1
					sum += 1
	print('a values ', 100 * (correct0 / sum), '% correct,', correct0, 'values wrong.\n')

	en_word_dict, tk_word_dict = dicts_for_train_comparison(src, tar)
	correct0 = 0
	sum = 0
	for sentence, srcs in enumerate(src):
		tars = tar[sentence]
		for index, eng_word in enumerate(srcs):  # for all words
			for index, tur_word in enumerate(tars):  # for all words
				idx_tur_in_dict = tk_word_dict[tur_word]
				idx_eng_in_dict = en_word_dict[eng_word]
				if idx_tur_in_dict < t_e_f_mat.shape[0] and idx_eng_in_dict < t_e_f_mat.shape[1]:
					val = t_e_f_mat[idx_tur_in_dict][idx_eng_in_dict]
					bool_ = (t[f_i][e_j] > 0.7) == (val > 0.7)
					if bool_ == False:
						if DEBUG: print('wrong a:', t[f_i][e_j], '!=', val, 'for i', i, ' j', j, ' l_e', l_e, ' l_f', l_f)
					else:
						correct0 += 1
				sum += 1
	print('t values ', 100 * (correct0 / sum), '% correct,', correct0, 'values wrong.\n')

# Alitalia  4647TL 28.6. - 31.7. IST
# Air China 4124TL 28.6. - 31.7. MUC

def is_converged2(new,old,num_of_iterations, max_it):
	epsilone = 0.00000001
	if num_of_iterations > max_it :
		return True

	for i in range(len(new)):
		for j in range(len(new[0])):
			if math.fabs(new[i][j]- old[i][j]) > epsilone:
				return False
	return True


### The following code is not mine and I only used it to test my code ###
# The author is https://github.com/onurmus/IBM-Models-1-2-3 #
# TODO remove
def train(t_e_f_mat, e_word_dict,f_word_dict,e_sentences,f_sentences,max_le,max_lf, max_it):
	a_i_le_lf_mat = np.zeros((max_lf, max_le, max_lf,max_le), dtype=float)
	if DEBUG: print('start train()')
	for lf in range(max_lf):
		a_i_le_lf_mat[:,:,lf,:] = 1/(lf+1)

	num_of_e_word = len(e_word_dict)
	num_of_f_word = len(f_word_dict)

	t_e_f_mat_prev = np.full((num_of_e_word, num_of_f_word), 1,dtype=float)
	cnt_iter = 0

	while not is_converged2(t_e_f_mat,t_e_f_mat_prev,cnt_iter, max_it):
		if DEBUG and cnt_iter % 10 == 0: print('step', cnt_iter,'of', max_it)
		cnt_iter += 1
		t_e_f_mat_prev = t_e_f_mat.copy()
		count_e_f = np.full((num_of_e_word, num_of_f_word), 0, dtype=float)
		total_f = np.full((num_of_f_word),0, dtype=float)
		count_a_i_le_lf = np.zeros((max_lf, max_le, max_lf,max_le), dtype=float)
		total_a_j_le_lf = np.zeros((max_le,max_le,max_lf),dtype=float)

		for idx_e, e_sen in enumerate(e_sentences): #for all sentence pairs (e,f) do
			#le = length(e), lf = length(f)
			e_sen_words = e_sen#.split(" ")
			f_sen_words = f_sentences[idx_e]#.split(" ")
			l_e = len(e_sen_words)
			l_f = len(f_sen_words)

			#compute normalization
			s_total = np.full((l_e),0,dtype=float)
			for j in range(l_e): #for j = 1 .. le do // all word positions in e
				s_total[j] = 0 #s-total(ej) = 0
				e_word = e_sen_words[j]
				for i in range(l_f): #for i = 0 .. lf do // all word positions in f
					f_word = f_sen_words[i]
					e_j = e_word_dict[e_word]
					f_i = f_word_dict[f_word]
					s_total[j] += t_e_f_mat[e_j][f_i] * a_i_le_lf_mat[i][j][l_f-1][l_e-1] #s-total(ej) += t(ej|fi) ∗ a(i|j,le,lf)
				#end for
			#end for

			#collect counts
			for j in range(l_e): #for j = 1 .. le do // all word positions in e
				e_word = e_sen_words[j]
				for i in range(l_f): #for i = 0 .. lf do // all word positions in f
					f_word = f_sen_words[i]
					e_j = e_word_dict[e_word]
					f_i = f_word_dict[f_word]

					c = t_e_f_mat[e_j][f_i] * a_i_le_lf_mat[i][j][l_f-1][l_e-1] / s_total[j] #c = t(ej|fi) ∗ a(i|j,le,lf) / s-total(ej)
					count_e_f[e_j][f_i] += c #count(ej|fi) += c
					total_f[f_i] += c #total(fi) += c
					count_a_i_le_lf[i][j][l_f-1][l_e-1] += c #counta(i|j,le,lf) += c
					total_a_j_le_lf[j][l_e-1][l_f-1] += c #totala(j,le,lf) += c
				# end for
			# end for
		# end for
		# estimate probabilities
		t_e_f_mat = np.full((num_of_e_word, num_of_f_word), 0, dtype=float)  # t(e|f) = 0 for all e,f
		a_i_le_lf_mat = np.zeros((max_lf, max_le, max_lf, max_le),
								 dtype=float)  # a(i|j,le,lf) = 0 for all i,j,le,lf
		for f_idx in range(num_of_f_word):  # for all foreign words f do
			for e_idx in range(num_of_e_word):  # for all English words e do
				if count_e_f[e_idx][f_idx] != 0:
					t_e_f_mat[e_idx][f_idx] = count_e_f[e_idx][f_idx] / total_f[f_idx]
			# end for
		# end for

		for i in range(max_lf):
			for j in range(max_le):
				for le in range(max_le):
					for lf in range(max_lf):
						if count_a_i_le_lf[i][j][lf][le] != 0:
							a_i_le_lf_mat[i][j][lf][le] = count_a_i_le_lf[i][j][lf][le] / total_a_j_le_lf[j][le][lf]
	if DEBUG: print('finish train()')
	return t_e_f_mat, a_i_le_lf_mat


# calculate sum of a over all target words: should be 1
def test_sum_a_is_one(a, tar):
	sum = 0
	(_, j, l_e, l_f), value = a.popitem()  # choose random entry
	for i in range(len(tar)):
		sum += a.get((i, j, l_e, l_f), 0)
	if sum < 0.99:
		print('Ohoh, sum is supposed to be 1, not', sum, ':(')
	else:
		print('Sum over all f is', sum, ':)')