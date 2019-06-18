from copy import copy

from src.evaluation import check_b_item_pp_at_end
from src.plotting import plot_cat_and_item_pps
from src.utils import print_params
from src.utils import calc_max_cat_pp
from src.jobs import train_loop
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config


# params
input_params = copy(config.Axb)
rnn_params = copy(config.RNN)
print_params(rnn_params)

# seqs_data
train_corpus = AxbCorpus(input_params, name='train')
test_corpus = AxbCorpus(input_params,  name='test')
master_vocab = Vocab(train_corpus, test_corpus)

# train
rnn = RNN(master_vocab, rnn_params)
corpus2results = train_loop(rnn, master_vocab)

# plot
max_cat_pp = calc_max_cat_pp(input_params, train_corpus.num_sequences, master_vocab.num_items)
plot_cat_and_item_pps(corpus2results, corpus_names=['train'], max_cat_pp=max_cat_pp)


check_b_item_pp_at_end(rnn, input_params, master_vocab, corpus2results)