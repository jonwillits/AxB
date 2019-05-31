import numpy as np

from src.evaluation import calc_pps
from src import config


def make_name2seqs(master_vocab, train_corpus, test_corpus):
    train_seqs = master_vocab.generate_index_sequences(train_corpus)
    test_seqs = master_vocab.generate_index_sequences(test_corpus)
    novel_seqs = [seq for seq in test_seqs if seq not in train_seqs] or test_seqs
    if config.Eval.skip_novel:
        res = {name: seqs for name, seqs in zip(('train', 'test'),
                                                (train_seqs, test_seqs))}
    else:
        res = {name: seqs for name, seqs in zip(('train', 'test', 'novel'),
                                                (train_seqs, test_seqs, novel_seqs))}
    return res


def train_loop(srn, input_params, name2seqs, master_vocab):
    # init
    train_seqs = name2seqs['train']
    seq_names = name2seqs.keys()
    distances = np.arange(1, config.Eval.max_distance + 1)
    name2dist2cat_pps = {name: {dist: [] for dist in distances} for name in seq_names}
    name2dist2type_pps = {name: {dist: [] for dist in distances} for name in seq_names}
    #
    seqs_pp = srn.train_epoch(train_seqs, train=False)
    #
    for epoch in range(srn.params.num_epochs):
        # cat_pp + type_pp
        assert input_params.punct  # required to calculate split_indices correctly
        split_indices = np.cumsum([1, input_params.num_ab_types,
                                   input_params.num_ab_types, input_params.num_x_test_types])
        calc_pps(srn, master_vocab, name2seqs, name2dist2cat_pps, name2dist2type_pps, split_indices)
        # train
        if config.Verbosity.seqs_pp:
            print('seqs_pp={}'.format(seqs_pp))
        seqs_pp = srn.train_epoch(train_seqs, train=True)

    return name2dist2cat_pps, name2dist2type_pps