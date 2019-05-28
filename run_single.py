from copy import copy

from src.plotting import plot_cat_and_type_pps
from src.utils import print_params
from src.utils import calc_max_cat_pp
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src import config


# params
input_params = copy(config.Input)
rnn_params = copy(config.RNN)
print_params(rnn_params)

# seqs_data
train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
master_vocab = Vocab(train_corpus, test_corpus)
name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)

# train
name2cat2cat_pps, name2cat2type_pps, b_type_pp_at_end = train_loop(
    rnn_params, input_params, name2seqs, master_vocab)

# plot
max_cat_pp = calc_max_cat_pp(input_params, train_corpus.num_sequences, master_vocab.master_vocab_size)
plot_cat_and_type_pps(name2cat2cat_pps, name2cat2type_pps, seq_names=['train'], max_cat_pp=max_cat_pp)