from copy import copy
import numpy as np

from src.evaluation import make_dist2type_pp_at_end
from src.utils import print_params
from src.plotting import plot_grid_mat
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

PARAMS1_NAME = 'learning_rate'
PARAMS1 = [0.1, 0.25]  # [0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [2, 8]  # [2, 3, 4, 5, 6, 7, 8]

MAX_NUM_EPOCHS = 50

config.General.type_pp_verbose = False


# params
input_params = copy(config.Input)
rnn_params = copy(config.RNN)

# seqs_data
train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
master_vocab = Vocab(train_corpus, test_corpus)
name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)

# grid search
distances = np.arange(input_params.min_distance, input_params.max_distance + 1)
dist2grid_mat = {dist: np.zeros((len(PARAMS1), len(PARAMS2))) for dist in distances}
for i, param1 in enumerate(PARAMS1):
    for j, param2 in enumerate(PARAMS2):
        # overwrite params
        setattr(rnn_params, PARAMS1_NAME, param1)
        setattr(rnn_params, PARAMS2_NAME, param2)
        setattr(rnn_params, 'num_epochs', MAX_NUM_EPOCHS)
        print_params(rnn_params)
        #
        for _ in range(config.General.num_reps):
            # train + calc pp
            srn = RNN(master_vocab.master_vocab_size, rnn_params)
            _, name2dist2type_pps = train_loop(srn, input_params, name2seqs, master_vocab)
            dist2type_pp_at_end = make_dist2type_pp_at_end(
                srn, input_params, master_vocab, name2seqs, name2dist2type_pps)

            # populate grid_mat
            for dist in distances:
                dist2grid_mat[dist][i, j] += dist2type_pp_at_end[dist] / config.General.num_reps

# plot  # TODO combine distances into one figure
plot_grid_mat(dist2grid_mat, MAX_NUM_EPOCHS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)