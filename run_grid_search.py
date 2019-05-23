

from src.utils import print_params
from src.jobs import train_loop
from src.jobs import make_seqs_data
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src import config


LEARNING_RATES = [0.001, 0.01, 0.1]
HIDDEN_SIZES = [2, 4, 8, 16, 32]

# seqs_data
input_params = config.Input
train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
master_vocab = Vocab(train_corpus, test_corpus)
seqs_data = make_seqs_data(master_vocab, train_corpus, test_corpus)


# grid search
rnn_params = config.RNN
for learning_rate in LEARNING_RATES:
    for hidden_size in HIDDEN_SIZES:
        rnn_params.hidden_size = hidden_size
        rnn_params.learning_rate = learning_rate
        print_params(rnn_params)
        _, _, is_success = train_loop(rnn_params, input_params, seqs_data, master_vocab)

        print('is_success={}'.format(is_success))
        # TODO keep track in mat of success

