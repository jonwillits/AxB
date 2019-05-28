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
PARAMS1 = [0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [2, 3, 4, 5, 6, 7, 8]

MAX_NUM_EPOCHS = 100

# params
input_params = copy(config.Input)
rnn_params = copy(config.RNN)

# seqs_data
train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
master_vocab = Vocab(train_corpus, test_corpus)
seqs_data = make_seqs_data(master_vocab, train_corpus, test_corpus)


# grid search
grid_mat = np.zeros((len(PARAMS1), len(PARAMS2)))
for i, param1 in enumerate(PARAMS1):
    for j, param2 in enumerate(PARAMS2):
        # overwrite params
        setattr(rnn_params, PARAMS1_NAME, param1)
        setattr(rnn_params, PARAMS2_NAME, param2)
        setattr(rnn_params, 'num_epochs', MAX_NUM_EPOCHS)
        print_params(rnn_params)
        #
        sum_epoch_before_conv = 0.0
        for _ in range(config.General.num_reps):
            _, _, epoch_before_conv = train_loop(rnn_params, input_params, seqs_data, master_vocab)
            sum_epoch_before_conv += epoch_before_conv
        avg_epoch_before_conv = sum_epoch_before_conv / config.General.num_reps
        print('avg_epoch_before_conv={}'.format(avg_epoch_before_conv))
        grid_mat[i, j] = avg_epoch_before_conv

# plot
plot_grid_mat(grid_mat, MAX_NUM_EPOCHS, PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)
