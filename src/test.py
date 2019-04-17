import nltk
from nltk.translate import AlignedSent, IBMModel1, IBMModel2

def initialize_test_sets():
	e_train = [['house','the','the'],['the','the','book'],['a','book']]
	f_train = [['das','Haus'],['das','Buch'],['ein','Buch']]
	e_test = [['the'],['the','the','house','a','book'],['book','a'],['book','the'],['kgefw', 'fek']]
	f_test = [['das'],['das','ein','Buch','Haus'],['Buch','ein'],['Buch','ein'],['Buch','ein']]
	all_alignments_test = {0:[[1]], 1:[[1,0,0,0],[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]], 2:[[1,0],[0,1]], 3:[[1,0],[0,1]], 4:[[1,0],[0,1]]}
	return e_train, f_train, e_test, f_test, all_alignments_test

def test_sets_to_aligned(src, tar):
	bitext = []
	for sentence in range (0, len(src)):
		bitext.append(AlignedSent(tar[sentence], src[sentence]))
	return bitext

def compare_t_table(ibm, t):
	for (e_j,f_i) in t:
		bool = (ibm.translation_table[f_i][e_j] > 0.7) == (t[(e_j,f_i)] > 0.7)
		if bool == False:
			print('wrong t', e_j, f_i, t[(e_j,f_i)], ibm.translation_table[f_i][e_j])

#compare t table to IBM Model 1 implementation of nltk library
def compare_ibm_1(t, max_steps, src, tar):
	aligned = test_sets_to_aligned(src, tar)
	ibm1 = IBMModel1(aligned, max_steps)
	compare_t_table(ibm1, t)

#compare t and a table to IBM Model 2 implementation of nltk library
def compare_ibm_2(t, max_steps, a, src, tar):
	aligned = test_sets_to_aligned(src, tar)
	ibm2 = IBMModel2(aligned, max_steps)
	compare_t_table(ibm2, t)
	e_lengths = [len(e) for e in src]
	f_lengths = [len(f) for f in tar]
	for (i,j,l_e,l_f) in a:
		bool = (a[(i,j,l_e,l_f)] > 0.7) == (ibm2.alignment_table[j][i][l_f][l_e] > 0.7)
		if bool == False:
			#print('wrong a', a[(i,j,l_e,l_f)], ibm2.alignment_table[j][i][l_f][l_e])
			print('wrong a','i',i,' j',j,' l_e',l_e,' l_f',l_f)
		#else:
		    #print('true','i',i,' j',j,' l_e',l_e,' l_f',l_f)

