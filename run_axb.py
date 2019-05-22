import numpy as np


from src.corpus import AxbCorpus
from src.rnn import RNN
from src.vocab import Vocab
from src.utils import evaluate
from src.utils import plot_pp_trajs


"""
note on perplexity (pp):
when 
NUM_AB_TYPES = 2
NUM_X_TRAIN_TYPES = 4
NUM_X_TEST_TYPES = 8
MAX_DISTANCE = 1
MIN_DISTANCE = 1
then, theoretical minimum should be 13/8=1.625, where 13=2+4+1+1 (sum of pps at each position)


initial perplexity for categories should be:
-np.log(1/master_vocab-size)*num_windows_where_target / num_windows
An example, given the hyper parameters above:
-np.log(1/13)*8 / 32 = 0.641
(8 is the number of times in the sequences where the answer is correct)
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

# sequences
master_vocab = Vocab(train_corpus, test_corpus)
train_seqs = master_vocab.generate_index_sequences(train_corpus)
test_seqs = master_vocab.generate_index_sequences(test_corpus)
novel_seqs = [seq for seq in test_seqs if seq not in train_seqs]
seq_names = ('train', 'test', 'novel')
seqs_data = list(zip((train_seqs, test_seqs, novel_seqs), seq_names))
print(master_vocab.master_vocab_list)

# check
avg_window_size = (1 if train_corpus.punct else 0) + 2 + np.mean([MAX_DISTANCE, MIN_DISTANCE])
num_windows = avg_window_size * train_corpus.num_sequences
max_pp = -np.log(1/master_vocab.master_vocab_size) * train_corpus.num_sequences / num_windows


# train + evaluate SRN
name2cat2cat_pps = {name: {cat: [] for cat in ['.', 'A', 'x', 'B']} for name in seq_names}
name2cat2type_pps = {name: {cat: [] for cat in ['.', 'A', 'x', 'B']} for name in seq_names}
srn = RNN(master_vocab)

print('{:13s} {:10s}{:10s}{:10s}{:10s}{:10s}'.format('Epoch', 'Seqs-PP', 'A', 'x', 'B', '.'))
for epoch in range(srn.epochs):
    # evaluate
    # TODO if params change, then split_indices must change
    assert NUM_AB_TYPES == 2
    assert NUM_X_TRAIN_TYPES == 4
    assert NUM_X_TEST_TYPES == 8
    evaluate(srn, master_vocab, seqs_data, name2cat2cat_pps, name2cat2type_pps)
    pp = srn.calc_seqs_pp(train_seqs)
    accuracies = srn.calc_accuracies(train_seqs, train_corpus)
    print('{:8}: {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}'.format(
        epoch, pp, accuracies[0], accuracies[1], accuracies[2], accuracies[3] if train_corpus.punct else np.nan))
    # train
    srn.train_epoch(train_seqs, VERBOSE, NUM_EVAL_STEPS)

# plot
for name in seq_names:
    plot_pp_trajs(name2cat2cat_pps[name], name, 'Category', y_max=max_pp)
for name in seq_names:
    plot_pp_trajs(name2cat2type_pps[name], name, 'Type', y_max=1.0)