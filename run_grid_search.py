from copy import copy
import numpy as np
from itertools import product
import datetime

from src.evaluation import make_name2dist2type_pp_at_end
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
DISTANCES = [[1, 1], [1, 2], [1, 3]]
MAX_NUM_EPOCHS = 100
PLOT_SEQ_NAMES = ['train', 'test']

config.General.type_pp_verbose = False


# params
input_params = copy(config.Input)
rnn_params = copy(config.RNN)

# do for each distance setting
for min_d, max_d in DISTANCES:
    setattr(input_params, 'min_distance', min_d)
    setattr(input_params, 'max_distance', max_d)

    # seqs_data
    train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
    test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
    master_vocab = Vocab(train_corpus, test_corpus)
    name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)
    seq_names = name2seqs.keys()

    # grid search over rnn_params
    distances = np.arange(input_params.min_distance, input_params.max_distance + 1)
    name2dist2grid_mat = {seq_name: {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
                          for seq_name in seq_names}
    for i, param1 in enumerate(PARAMS1):
        for j, param2 in enumerate(PARAMS2):
            # overwrite params
            setattr(rnn_params, PARAMS1_NAME, param1)
            setattr(rnn_params, PARAMS2_NAME, param2)
            setattr(rnn_params, 'num_epochs', MAX_NUM_EPOCHS)
            setattr(rnn_params, 'bptt', max_d + 1)
            print_params(rnn_params)
            #
            for _ in range(config.General.num_reps):
                # train + calc pp
                srn = RNN(master_vocab.master_vocab_size, rnn_params)
                _, name2dist2type_pps = train_loop(srn, input_params, name2seqs, master_vocab)
                name2dist2type_pp_at_end = make_name2dist2type_pp_at_end(
                    srn, input_params, master_vocab, name2seqs, name2dist2type_pps)

                # populate grid_mat
                for seq_name, dist in product(seq_names, distances):
                    name2dist2grid_mat[seq_name][dist][i, j] += \
                        name2dist2type_pp_at_end[seq_name][dist] / config.General.num_reps

    # plot
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    plot_grid_search_results(time_stamp, name2dist2grid_mat, PLOT_SEQ_NAMES,
                             MAX_NUM_EPOCHS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)
    plot_params(time_stamp, input_params, rnn_params)