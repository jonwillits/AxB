import numpy as np
import datetime
import sys
import pyprind
from itertools import product

from src.evaluation import check_item_pp_at_end
from src.utils import print_params
from src.plotting import plot_grid_search_results
from src.plotting import plot_params
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

PARAMS1_NAME = 'learning_rate'
PARAMS1 = [0.25] or [0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [8] or [2, 4, 6, 8]
TRAIN_DISTANCES = [[0, 1]]
TRAIN_X_SET_SIZES = [2, 4, 6]
MAX_NUM_EPOCHS = 100
PLOT_SEQ_NAMES = ['train', 'test']
NUM_REPS = 1
PROGRESS_BAR = True
LIMIT_BPPT = False  # if True, generalization to unseen distances is impossible


# params
input_params = config.Axb  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt such that it is possible to learn dependencies across largest distance
setattr(rnn_params, 'bptt', config.Eval.max_distance + 1)
print('Set bptt to {}'.format(rnn_params.bptt))

# do for each distance setting
for (min_d, max_d), train_x_cat_size in product(TRAIN_DISTANCES, TRAIN_X_SET_SIZES):

    # progressbar
    print('Grid search with min_distance={} and max_distance={}'.format(min_d, max_d))
    pbar = pyprind.ProgBar(len(PARAMS1) * len(PARAMS2), stream=sys.stdout)

    # modify input_params before generating sequences
    setattr(input_params, 'min_distance', min_d)
    setattr(input_params, 'max_distance', max_d)
    setattr(input_params, 'train_x_cat_size', train_x_cat_size)

    if LIMIT_BPPT:  # sets bptt to maximal bptt needed to learn training dependencies only
        setattr(rnn_params, 'bptt', max_d + 1)
        print('Set bptt to {}'.format(rnn_params.bptt))

    # make train and test sequences
    train_corpus = AxbCorpus(input_params, test=False)
    test_corpus = AxbCorpus(input_params,  test=True)
    master_vocab = Vocab(train_corpus, test_corpus)
    name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)
    seq_names = name2seqs.keys()

    # init result data structures
    distances = np.arange(1, config.Eval.max_distance + 1)
    name2dist2item_pp_mat = {seq_name: {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
                             for seq_name in seq_names}
    name2dist2cat_pp_mat = {seq_name: {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
                            for seq_name in seq_names}
    name2dist2item_pp_start = {seq_name: {dist: None for dist in distances}
                               for seq_name in seq_names}
    name2dist2cat_pp_start = {seq_name: {dist: None for dist in distances}
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
                name2dist2cat_pps, name2dist2item_pps = cat2results['B']

                # check item-perplexity against theory
                if not PROGRESS_BAR:
                    check_item_pp_at_end(
                        rnn, input_params, master_vocab, name2seqs, name2dist2item_pps)

                # populate result data structures
                for seq_name, dist in product(seq_names, distances):
                    item_pps = name2dist2item_pps[seq_name][dist]
                    cat_pps = name2dist2cat_pps[seq_name][dist]
                    if not item_pps or not cat_pps:
                        continue
                    # category-perplexity
                    name2dist2cat_pp_mat[seq_name][dist][i, j] += cat_pps[-1] / NUM_REPS
                    name2dist2cat_pp_start[seq_name][dist] = cat_pps[0]
                    # item-perplexity
                    name2dist2item_pp_mat[seq_name][dist][i, j] += item_pps[-1] / NUM_REPS
                    name2dist2item_pp_start[seq_name][dist] = item_pps[0]

            if PROGRESS_BAR:
                pbar.update()

    # plot heatmaps showing item or category perplexity for al hyper-parameter configurations
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    setattr(rnn_params, PARAMS1_NAME, '<grid_search>')
    setattr(rnn_params, PARAMS2_NAME, '<grid_search>')
    plot_params(time_stamp, input_params, rnn_params)
    plot_grid_search_results(time_stamp, 'B', 'Item', name2dist2item_pp_mat, name2dist2item_pp_start, PLOT_SEQ_NAMES,
                             MAX_NUM_EPOCHS, NUM_REPS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)
    plot_grid_search_results(time_stamp, 'B', 'Category', name2dist2cat_pp_mat, name2dist2cat_pp_start, PLOT_SEQ_NAMES,
                             MAX_NUM_EPOCHS, NUM_REPS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)


