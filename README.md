# Machine translation

A python library for machine translation. For now there are implementations using the IBM Model 1, IBM Model 2 and a phrase-based model.

## Usage and Examples

To train on a corpus, the corpus file has to be read in. The methods read_in_corpus(), split_dataset() and initialize_test_sets() should be adapted to fit the filestructure of the corpus file. It is assumed the corpus consists of 2 files (one for source language, the other one for target language) each with one sentence per line.

### IBM Model 1

The method EM_IBM_Model_1() in ibm1.py has to be called to train the model on the corpus. 
```
t_s2t_ibm1 = EM_IBM_Model_1(s_train, t_train, max_steps)
```
Then the probability of a translation given a source can be calculated.
```
p = prob_e_given_f_1(s, t, 1, t_s2t_ibm1)
```

### IBM Model 2

The method EM_IBM_Model_2() in ibm2.py has to be called to train the model on the corpus. For that first IBM model 1 has to be trained. 
```
t_s2t_ibm1 = EM_IBM_Model_1(s_train, t_train, max_steps)
t_s2t_ibm2, a_s2t_ibm2 = EM_IBM_Model_2(s_train, t_train, t_s2t_ibm1, max_steps)
```
Then the probability of a translation given a source can be calculated.
```
p = prob_e_given_f_2(s, t, 1, t_s2t_ibm2, a_s2t_ibm2)
```

### Phrase-based model

To use this model the words in the source sentences need to be aligned with the words in the target sentences. IBM model 2 is used for this word alignment.
 ```
t_s2t_ibm1 = EM_IBM_Model_1(s_train, t_train, max_steps)
t_s2t_ibm2, a_s2t_ibm2 = EM_IBM_Model_2(s_train, t_train, t_s2t_ibm1, max_steps)
t_t2s_ibm1 = EM_IBM_Model_1(t_train, s_train, max_steps)
t_t2s_ibm2, a_t2s_ibm2 = EM_IBM_Model_2(t_train, s_train, t_t2s_ibm1, max_steps)
all_a_train = word_alignment(s_train, t_train, t_s2t_ibm2, a_s2t_ibm2, t_t2s_ibm2, a_t2s_ibm2)
```
Phrases are extracted and phrase translation probabilities estimated.
```
phrase_counts = phrase_extraction(s_train, t_train, all_a_train, max_phrase_len)
```
The bigram language model is trained.
```
unigram_counts, bigram_counts = count_grams(s_train)
```
Then the probability of a translation given a source can be calculated, where phrases_a is an alignment of the phrases in format {s_to: (s_from, t_from, t_to)}.
```
p = prob_e_given_f(s, t, phrase_counts, unigram_counts, bigram_counts, phrases_a)
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

I want to mention, that I used the book ['Statistical Machine Translation'](https://www.cambridge.org/core/books/statistical-machine-translation/94EADF9F680558E13BE759997553CDE5) by Philipp Koehn as a basis for my code.
