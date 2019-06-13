import numpy as np
import datetime
import sys
import pyprind
from itertools import product

from src.evaluation import check_b_item_pp_at_end
from src.utils import print_params
from src.plotting import plot_grid_search_results
from src.plotting import plot_params
from src.jobs import train_loop
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

PARAMS1_NAME = 'learning_rate'
PARAMS1 = [0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [2, 4, 6, 8]
TRAIN_DISTANCES = [[0, 1]]
MAX_EVAL_DISTANCE = 3
TRAIN_X_SET_SIZES = [2, 4, 6]
MAX_NUM_EPOCHS = 10
NUM_REPS = 1
PROGRESS_BAR = True
LIMIT_BPPT = False  # if True, generalization to unseen distances is impossible


# params
input_params = config.Axb  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt such that it is possible to learn dependencies across largest distance
setattr(rnn_params, 'bptt', config.Eval.max_distance + 1)
print('Set bptt to {}'.format(rnn_params.bptt))

distances = np.arange(config.Eval.max_distance + 1)

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
    train_corpus = AxbCorpus(input_params, name='train')
    test_corpus = AxbCorpus(input_params,  name='test')
    master_vocab = Vocab(train_corpus, test_corpus)

    # init result data structures
    name2dist2item_pp_mat = {corpus.name: {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
                             for corpus in master_vocab.corpora}
    name2dist2cat_pp_mat = {corpus.name: {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
                            for corpus in master_vocab.corpora}
    name2dist2item_pp_start = {corpus.name: {dist: None for dist in distances}
                               for corpus in master_vocab.corpora}
    name2dist2cat_pp_start = {corpus.name: {dist: None for dist in distances}
                              for corpus in master_vocab.corpora}

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
                corpus2results = train_loop(rnn, master_vocab, distances)

                # check item-perplexity against theory
                if not PROGRESS_BAR:
                    check_b_item_pp_at_end(rnn, input_params, master_vocab, corpus2results)

                # populate result data structures
                for corpus in master_vocab.corpora:
                    for pos in corpus.positions:
                        dist = pos - 1  # e.g. B in A-x-B is in pos=2 and dist=1
                        if dist < 0:  # impossible for AxB corpus
                            continue
                        #
                        cat_pps = corpus2results[corpus.name]['B'][pos]['cat_pp']
                        item_pps = corpus2results[corpus.name]['B'][pos]['item_pp']
                        if not item_pps or not cat_pps:
                            continue
                        # category-perplexity
                        name2dist2cat_pp_mat[corpus.name][dist][i, j] += cat_pps[-1] / NUM_REPS
                        name2dist2cat_pp_start[corpus.name][dist] = cat_pps[0]
                        # item-perplexity
                        name2dist2item_pp_mat[corpus.name][dist][i, j] += item_pps[-1] / NUM_REPS
                        name2dist2item_pp_start[corpus.name][dist] = item_pps[0]

            if PROGRESS_BAR:
                pbar.update()

    # plot heatmaps showing item or category perplexity for al hyper-parameter configurations
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    seq_names = [corpus.name for corpus in master_vocab.corpora]
    setattr(rnn_params, PARAMS1_NAME, '<grid_search>')
    setattr(rnn_params, PARAMS2_NAME, '<grid_search>')
    plot_params(time_stamp, input_params, rnn_params)
    plot_grid_search_results(time_stamp, 'B', 'Category', name2dist2cat_pp_mat, name2dist2cat_pp_start, seq_names,
                             MAX_NUM_EPOCHS, NUM_REPS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)
    plot_grid_search_results(time_stamp, 'B', 'Item', name2dist2item_pp_mat, name2dist2item_pp_start, seq_names,
                             MAX_NUM_EPOCHS, NUM_REPS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)


