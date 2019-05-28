import numpy as np

from src.evaluation import calc_pps
from src.rnn import RNN
from src import config


def make_name2seqs(master_vocab, train_corpus, test_corpus):
    train_seqs = master_vocab.generate_index_sequences(train_corpus)
    test_seqs = master_vocab.generate_index_sequences(test_corpus)
    novel_seqs = [seq for seq in test_seqs if seq not in train_seqs] or test_seqs
    res = {name: seqs for name, seqs in zip(('train', 'test', 'novel'), (train_seqs, test_seqs, novel_seqs))}

    # TODO debug
    # for seq in train_seqs:
    #     print(seq)
    # raise SystemExit
    #

    return res


def train_loop(rnn_params, input_params, name2seqs, master_vocab):
    # init
    train_seqs = name2seqs['train']
    seq_names = name2seqs.keys()
    name2cat2cat_pps = {name: {cat: [] for cat in ['.', 'A', 'x', 'B']} for name in seq_names}
    name2cat2type_pps = {name: {cat: [] for cat in ['.', 'A', 'x', 'B']} for name in seq_names}
    srn = RNN(master_vocab.master_vocab_size, rnn_params)
    #
    seqs_pp = srn.train_epoch(train_seqs, train=False)
    #
    for epoch in range(srn.params.num_epochs):
        # cat_pp + type_pp
        assert input_params.punct  # required to calculate split_indices correctly
        split_indices = np.cumsum([1, input_params.num_ab_types,
                                   input_params.num_ab_types, input_params.num_x_test_types])
        calc_pps(srn, master_vocab, train_seqs, 'train', name2cat2cat_pps, name2cat2type_pps, split_indices)
        # train
        print('seqs_pp={}'.format(seqs_pp))
        seqs_pp = srn.train_epoch(train_seqs, train=True)

    #
    # TODO hand calculate
    x, y = srn.to_x_and_y(train_seqs)
    num_windows = len(y)
    max_b_type_pp = -np.log(1 / input_params.num_ab_types) * len(train_seqs) / num_windows
    b_type_pp_at_start = name2cat2type_pps['train']['B'][0]
    b_type_pp_at_end = name2cat2type_pps['train']['B'][-1]
    print()
    print('max_b_type_pp', max_b_type_pp)
    print('b_type_pp_at_start={}'.format(b_type_pp_at_start))
    print('b_type_pp_at_end={}'.format(b_type_pp_at_end))

    return name2cat2cat_pps, name2cat2type_pps, b_type_pp_at_end