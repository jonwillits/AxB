from copy import copy
import numpy as np

from src.plotting import plot_cat_and_type_pps
from src.utils import print_params
from src.utils import calc_max_cat_pp
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
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
srn = RNN(master_vocab.master_vocab_size, rnn_params)
name2dist2cat_pps, name2dist2type_pps = train_loop(
    srn, input_params, name2seqs, master_vocab)

# plot
max_cat_pp = calc_max_cat_pp(input_params, train_corpus.num_sequences, master_vocab.master_vocab_size)
plot_cat_and_type_pps(name2dist2cat_pps, name2dist2type_pps, seq_names=['train'], max_cat_pp=max_cat_pp)

# calculate theoretical maximum and minimum perplexity
# "B" type perplexity should converge on 1.0, even with variable size distance
distances = np.arange(input_params.min_distance, input_params.max_distance + 1)
for dist in distances:

    # TODO take into consideration the distance
    filtered_seqs = [seq for seq in name2seqs['train'] if len(seq) == 3 + dist]
    x, y = srn.to_x_and_y(filtered_seqs)
    num_windows = len(y)

    num_b_windows = np.sum([1 if master_vocab.types[yi].startswith('B') else 0 for yi in y])
    max_b_type_pp = np.exp(-np.log(1 / input_params.num_ab_types) * num_b_windows / num_windows)
    min_b_type_pp = np.exp(-np.log(1 / 1) * num_b_windows / num_windows)
    #
    b_type_pp_at_start = name2dist2type_pps['train'][dist][0]
    b_type_pp_at_end = name2dist2type_pps['train'][dist][-1]
    print('-------------')
    print('distance={}'.format(dist))
    print('-------------')
    print('num_b_windows', num_b_windows)
    print('num_windows', num_windows)
    print('max_b_type_pp', max_b_type_pp)
    print('min_b_type_pp', min_b_type_pp)
    print('b_type_pp_at_start={}'.format(b_type_pp_at_start))
    print('b_type_pp_at_end  ={}'.format(b_type_pp_at_end))