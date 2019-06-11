import numpy as np
from sortedcontainers import SortedDict

from src.evaluation import update_cat_and_item_pps
from src import config


def make_name2seqs(master_vocab, train_corpus, test_corpus=None):
    train_seqs = master_vocab.generate_index_sequences(train_corpus)
    test_seqs = master_vocab.generate_index_sequences(test_corpus)
    novel_seqs = [seq for seq in test_seqs if seq not in train_seqs] or test_seqs
    if config.Eval.skip_novel:
        res = {name: seqs for name, seqs in zip(('train', 'test'),
                                                (train_seqs, test_seqs))}
    else:
        res = {name: seqs for name, seqs in zip(('train', 'test', 'novel'),
                                                (train_seqs, test_seqs, novel_seqs))}
    return SortedDict(res)


def train_loop(srn, name2seqs, master_vocab, distances=None, eval_a=True):
    # init
    train_seqs = name2seqs['train']
    seq_names = name2seqs.keys()
    if distances is not None:  # training on AxB
        # each results object is a tuple: (name2dist2cat_pps, name2dist2item_pps)
        cat2results = {cat: ({name: {dist: [] for dist in distances} for name in seq_names},
                             {name: {dist: [] for dist in distances} for name in seq_names})
                       for cat in ['A', 'B']}
    else:   # training on Marcus
        # each results object is a tuple: (name2cat_pps, name2item_pps)
        cat2results = {cat: ({name: [] for name in seq_names},
                             {name: [] for name in seq_names})
                       for cat in ['A', 'B']}
    # calc seqs_pp + item_pp + cat_pp before training
    seqs_pp = srn.train_epoch(train_seqs, train=False)  # evaluate seqs_pp before training
    update_cat_and_item_pps(srn, master_vocab, name2seqs, 'A', distances, cat2results['A']) if eval_a else None
    update_cat_and_item_pps(srn, master_vocab, name2seqs, 'B', distances, cat2results['B'])
    # train + eval loop
    for epoch in range(srn.params.num_epochs):
        # train
        if config.Verbosity.seqs_pp:
            print('seqs_pp={}'.format(seqs_pp))
        seqs_pp = srn.train_epoch(train_seqs, train=True)
        # eval
        update_cat_and_item_pps(srn, master_vocab, name2seqs, 'A', distances, cat2results['A']) if eval_a else None
        update_cat_and_item_pps(srn, master_vocab, name2seqs, 'B', distances, cat2results['B'])
    return cat2results
