from copy import copy

from src.evaluation import check_item_pp_at_end
from src.plotting import plot_cat_and_item_pps
from src.utils import print_params
from src.utils import calc_max_cat_pp
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config


# params
input_params = copy(config.Axb)
rnn_params = copy(config.RNN)
print_params(rnn_params)

# seqs_data
train_corpus = AxbCorpus(input_params, test=False)
test_corpus = AxbCorpus(input_params,  test=True)
master_vocab = Vocab(train_corpus, test_corpus)
name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)

# train
rnn = RNN(master_vocab, rnn_params)
name2dist2cat_pps, name2dist2item_pps = train_loop(rnn, name2seqs, master_vocab)

# plot
max_cat_pp = calc_max_cat_pp(input_params, train_corpus.num_sequences, master_vocab.num_items)
plot_cat_and_item_pps(name2dist2cat_pps, name2dist2item_pps, seq_names=['train'], max_cat_pp=max_cat_pp)


check_item_pp_at_end(rnn, input_params, master_vocab, name2seqs, name2dist2item_pps)