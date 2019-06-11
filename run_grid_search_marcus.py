import numpy as np
import datetime
import sys
import pyprind
from itertools import product

from src.utils import print_params
from src.plotting import plot_grid_search_results_marcus
from src.plotting import plot_params
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import MarcusCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

PARAMS1_NAME = 'learning_rate'
PARAMS1 = [0.25] or [0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [8] or [2, 4, 6, 8]
MAX_NUM_EPOCHS = 100
PLOT_SEQ_NAMES = ['train', 'novel']
NUM_REPS = 1
PROGRESS_BAR = True

config.Eval.skip_novel = False  # evaluate on 'novel' sequences only (only unseen sequences)
config.Eval.max_distance = 1  # TODO what are the consequences of this?

# params
input_params = config.Marcus  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt
setattr(rnn_params, 'bptt', int(input_params.punctuation) * 2 + 2)  # TODO test
print('Set bptt to {}'.format(rnn_params.bptt))

# do for each marcus corpus pattern
for pattern in ['abb', 'aab', 'aba']:

    # progressbar
    print('Grid search with pattern={}'.format(pattern))
    pbar = pyprind.ProgBar(len(PARAMS1) * len(PARAMS2), stream=sys.stdout)

    # modify input_params before generating sequences
    setattr(input_params, 'pattern', pattern)

    # make train and test sequences
    train_corpus = MarcusCorpus(input_params, test=False)
    test_corpus = MarcusCorpus(input_params,  test=True)
    master_vocab = Vocab(train_corpus, test_corpus)
    name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)
    seq_names = name2seqs.keys()

    # init result data structures
    cats = ['A', 'B']
    name2cat2item_pp_mat = {seq_name: {cat: np.zeros((len(PARAMS1), len(PARAMS2))) for cat in cats}
                            for seq_name in seq_names}
    name2cat2cat_pp_mat = {seq_name: {cat: np.zeros((len(PARAMS1), len(PARAMS2))) for cat in cats}
                           for seq_name in seq_names}
    name2cat2item_pp_start = {seq_name: {cat: None for cat in cats}
                              for seq_name in seq_names}
    name2cat2cat_pp_start = {seq_name: {cat: None for cat in cats}
                             for seq_name in seq_names}

    # grid search over rnn_params
    for i, param1 in enumerate(PARAMS1):
        for j, param2 in enumerate(PARAMS2):

            # overwrite rnn_params
            setattr(rnn_params, PARAMS1_NAME, param1)
            setattr(rnn_params, PARAMS2_NAME, param2)
            setattr(rnn_params, 'num_epochs', MAX_NUM_EPOCHS)

            if not PROGRESS_BAR:
                print_params(rnn_params)

            # train and evaluate multiple models per hyper-parameter configuration
            for _ in range(NUM_REPS):

                # train + evaluate
                rnn = RNN(master_vocab, rnn_params)
                cat2results = train_loop(rnn, name2seqs, master_vocab)

                # populate result data structures
                for seq_name, cat in product(seq_names, cats):
                    name2dist2cat_pps, name2dist2item_pps = cat2results[cat]
                    cat_pps = name2dist2cat_pps[seq_name][1]
                    item_pps = name2dist2item_pps[seq_name][1]  # TODO what the consequences of dist=1 here?
                    if not item_pps or not cat_pps:
                        continue
                    # category-perplexity
                    name2cat2cat_pp_mat[seq_name][cat][i, j] += cat_pps[-1] / NUM_REPS
                    name2cat2cat_pp_start[seq_name][cat] = cat_pps[0]
                    # item-perplexity
                    name2cat2item_pp_mat[seq_name][cat][i, j] += item_pps[-1] / NUM_REPS
                    name2cat2item_pp_start[seq_name][cat] = item_pps[0]

            if PROGRESS_BAR:
                pbar.update()

    # plot heatmaps showing item or category perplexity for all hyper-parameter configurations
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    setattr(rnn_params, PARAMS1_NAME, '<grid_search>')
    setattr(rnn_params, PARAMS2_NAME, '<grid_search>')
    plot_params(time_stamp, input_params, rnn_params)
    plot_grid_search_results_marcus(time_stamp, 'Item', name2cat2item_pp_mat, name2cat2item_pp_start, pattern,
                                    PLOT_SEQ_NAMES, MAX_NUM_EPOCHS, NUM_REPS,
                                    PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)
    plot_grid_search_results_marcus(time_stamp, 'Category', name2cat2cat_pp_mat, name2cat2cat_pp_start, pattern,
                                    PLOT_SEQ_NAMES, MAX_NUM_EPOCHS, NUM_REPS,
                                    PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)


