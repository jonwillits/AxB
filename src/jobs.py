import numpy as np

from src.evaluation import calc_pps, calc_accuracies
from src.rnn import RNN
from src import config


def make_seqs_data(master_vocab, train_corpus, test_corpus):
    train_seqs = master_vocab.generate_index_sequences(train_corpus)
    test_seqs = master_vocab.generate_index_sequences(test_corpus)
    novel_seqs = [seq for seq in test_seqs if seq not in train_seqs] or test_seqs
    seqs_data = ((train_seqs, test_seqs, novel_seqs), ('train', 'test', 'novel'))
    return seqs_data


def train_loop(rnn_params, input_params, seqs_data, master_vocab, eval_pps=False):
    # init
    (train_seqs, test_seqs, novel_seqs), seq_names = seqs_data
    name2cat2cat_pps = {name: {cat: [] for cat in ['.', 'A', 'x', 'B']} for name in seq_names}
    name2cat2type_pps = {name: {cat: [] for cat in ['.', 'A', 'x', 'B']} for name in seq_names}
    srn = RNN(master_vocab.master_vocab_size, rnn_params)
    #
    seqs_pp = srn.train_epoch(train_seqs, train=False)
    accuracies = calc_accuracies(srn, train_seqs, master_vocab)
    num_b_successes = 0
    #
    if config.General.accuracies_verbose:
        print('{:13s} {:10s}{:10s}{:10s}{:10s}{:10s}'.format('Epoch', 'Seqs-PP', 'A', 'x', 'B', '.'))
    for epoch in range(srn.params.epochs):
        # cat_pp + type_pp
        if eval_pps:
            assert input_params.punct  # required to calculate split_indices correctly
            split_indices = np.cumsum([1, input_params.num_ab_types,
                                       input_params.num_ab_types, input_params.num_x_test_types])
            calc_pps(srn, master_vocab, seqs_data, name2cat2cat_pps, name2cat2type_pps, split_indices)
        # accuracy
        accuracies = calc_accuracies(srn, train_seqs, master_vocab)
        if config.General.accuracies_verbose:
            print('{:8}: {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}'.format(
                epoch, seqs_pp, accuracies[0], accuracies[1], accuracies[2],
                accuracies[3] if input_params.punct else np.nan))
        # train
        seqs_pp = srn.train_epoch(train_seqs, train=True)
        # success
        num_b_successes += 1 if accuracies[2] == 1.0 else -num_b_successes
        if num_b_successes >= config.General.success_num_epochs:
            return name2cat2cat_pps, name2cat2type_pps, True
    else:
        return name2cat2cat_pps, name2cat2type_pps, False