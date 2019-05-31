from copy import deepcopy
import numpy as np
from itertools import product
import datetime
import sys
import pyprind

from src.evaluation import check_type_pp_at_end
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
PARAMS1 = [0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [2, 4, 6, 8]
TRAIN_DISTANCES = [[1, 1], [1, 2], [1, 3]]
MAX_NUM_EPOCHS = 100
PLOT_SEQ_NAMES = ['train', 'test']
NUM_REPS = 1
PROGRESS_BAR = True

config.Verbosity.type_pp = False


# params
input_params = config.Input  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt such that it is possible to learn dependencies across largest distance
setattr(rnn_params, 'bptt', config.Eval.max_distance + 1)
print('Set bptt to {}'.format(rnn_params.bptt))

# do for each distance setting
for min_d, max_d in TRAIN_DISTANCES:

    # progressbar
    print('Grid search with min_distance={} and max_distance={}'.format(min_d, max_d))
    pbar = pyprind.ProgBar(len(PARAMS1) * len(PARAMS2), stream=sys.stdout)

    # set min and max distance before generating sequences
    setattr(input_params, 'min_distance', min_d)
    setattr(input_params, 'max_distance', max_d)

    # seqs_data
    train_corpus = AxbCorpus(input_params, test=False)
    test_corpus = AxbCorpus(input_params,  test=True)
    master_vocab = Vocab(train_corpus, test_corpus)
    name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)
    seq_names = name2seqs.keys()

    # init results
    distances = np.arange(1, config.Eval.max_distance + 1)
    name2dist2grid_mat = {seq_name: {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
                          for seq_name in seq_names}
    name2dist2start_pp = {seq_name: {dist: None for dist in distances}
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

            # each cell in grid is average over multiple models
            for _ in range(NUM_REPS):

                # train + evaluate
                rnn = RNN(master_vocab.num_types, master_vocab.types.index('PAD'), rnn_params)
                name2dist2cat_pps, name2dist2type_pps = train_loop(rnn, input_params, name2seqs, master_vocab)

                # check type-perplexity against theory
                if not PROGRESS_BAR:
                    check_type_pp_at_end(
                        rnn, input_params, master_vocab, name2seqs, name2dist2type_pps)

                # populate grid_mat
                for seq_name, dist2type_pps in name2dist2type_pps.items():
                    for dist, type_pps in dist2type_pps.items():
                        if not type_pps:
                            continue
                        name2dist2grid_mat[seq_name][dist][i, j] += type_pps[-1] / NUM_REPS
                        name2dist2start_pp[seq_name][dist] = type_pps[0]

            if PROGRESS_BAR:
                pbar.update()

    # plot
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    setattr(rnn_params, PARAMS1_NAME, '<grid_search>')
    setattr(rnn_params, PARAMS2_NAME, '<grid_search>')
    plot_params(time_stamp, input_params, rnn_params)
    plot_grid_search_results(time_stamp, name2dist2grid_mat, name2dist2start_pp, PLOT_SEQ_NAMES,
                             MAX_NUM_EPOCHS, NUM_REPS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)


