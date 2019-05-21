import numpy as np
import pandas as pd
from scipy.special import softmax

from src.corpus import AxbCorpus
from src.rnn import RNN
from src.vocab import Vocab


"""
note on perplexity (pp):
when 
NUM_AB_TYPES = 2
NUM_X_TRAIN_TYPES = 4
NUM_X_TEST_TYPES = 8
MAX_DISTANCE = 1
MIN_DISTANCE = 1
then, theoretical minimum should be 13/8=1.625, where 13=2+4+1+1 (sum of pps at each position)

"""


NUM_AB_TYPES = 2
NUM_X_TRAIN_TYPES = 4
NUM_X_TEST_TYPES = 8
MAX_DISTANCE = 1
MIN_DISTANCE = 1

NUM_EVAL_STEPS = 10
VERBOSE = False

# corpora
train_corpus = AxbCorpus(NUM_AB_TYPES, NUM_X_TRAIN_TYPES, MAX_DISTANCE, MIN_DISTANCE)
test_corpus = AxbCorpus(NUM_AB_TYPES, NUM_X_TEST_TYPES, MAX_DISTANCE, MIN_DISTANCE)

# train sequences
master_vocab = Vocab(train_corpus, test_corpus)
train_seqs = master_vocab.generate_index_sequences(train_corpus)
print(master_vocab.master_vocab_size)
for seq in train_seqs:
    print(seq)

# train SRN
srn = RNN(master_vocab)
print('{:13s} {:10s}{:10s}{:10s}{:10s}{:10s}'.format('Epoch', 'Seqs-PP', 'A', 'x', 'B', '.'))
for epoch in range(srn.epochs):
    # evaluate
    pp = srn.calc_seqs_pp(train_seqs)
    accuracies = srn.calc_accuracies(train_seqs, train_corpus)
    print('{:8}: {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}'.format(
        epoch, pp, accuracies[0], accuracies[1], accuracies[2], accuracies[3] if train_corpus.punct else np.nan))
    # train
    srn.train_epoch(train_seqs, VERBOSE, NUM_EVAL_STEPS)

# evaluation
all_windows = np.vstack([srn.to_windows(seq) for seq in train_seqs])
y = all_windows[:, -1]
logits = srn.calc_logits(train_seqs)
print(softmax(logits, axis=1).round(2))