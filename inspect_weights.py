from copy import copy

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
setattr(rnn_params, 'learning_rate', 0.25)


# seqs_data
train_corpus = AxbCorpus(input_params, num_x_types=input_params.num_x_train_types)
test_corpus = AxbCorpus(input_params,  num_x_types=input_params.num_x_test_types)
master_vocab = Vocab(train_corpus, test_corpus)
name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)

for hidden_size in [3, 4, 5, 6, 7, 8]:
    setattr(input_params, 'hidden_size', hidden_size)
    print_params(rnn_params)

    # train
    srn = RNN(master_vocab.master_vocab_size, rnn_params)
    name2dist2cat_pps, name2dist2type_pps = train_loop(
        srn, input_params, name2seqs, master_vocab)

    # plot type perplexity (to verify training was successful)
    max_cat_pp = calc_max_cat_pp(input_params, train_corpus.num_sequences, master_vocab.master_vocab_size)
    plot_cat_and_type_pps(name2dist2cat_pps, name2dist2type_pps, seq_names=['train'], max_cat_pp=max_cat_pp)

    # store weights in dict where the key is the name of the weight matrix
    name2array = {}
    for name, param in srn.model.named_parameters():
        name2array[name] = param.detach().numpy()

    # inspect contents of dict
    for name, weights in name2array.items():
        print(name)
        print(weights.shape)

    # wx: onehot-to-embedding
    # rnn.weight_ih_l0: embedding-to-hidden
    # wy: hidden-to-output
