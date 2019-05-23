from copy import copy
import numpy as np

from src.utils import print_params
from src.plotting import plot_grid_mat
from src.jobs import train_loop
from src.jobs import make_seqs_data
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src import config

PARAMS1_NAME = 'learning_rate'
PARAMS1 = [0.001, 0.01, 0.1, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [2, 4, 8, 16, 32]

# params
input_params = copy(config.Input)
rnn_params = copy(config.RNN)

# seqs_data
train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
master_vocab = Vocab(train_corpus, test_corpus)
seqs_data = make_seqs_data(master_vocab, train_corpus, test_corpus)


# grid search
grid_mat = np.zeros((len(PARAMS1), len(PARAMS2))).astype(np.bool)
for i, param1 in enumerate(PARAMS1):
    for j, param2 in enumerate(PARAMS2):
        setattr(rnn_params, PARAMS1_NAME, param1)
        setattr(rnn_params, PARAMS2_NAME, param2)
        print_params(rnn_params)
        _, _, is_success = train_loop(rnn_params, input_params, seqs_data, master_vocab)

        print('is_success={}'.format(is_success))
        grid_mat[i, j] = is_success

# plot  # TODO combine multiple mats for different num_epochs
plot_grid_mat(grid_mat, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)