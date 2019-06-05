import numpy as np
import datetime
import sys
import pyprind
from itertools import product

from src.evaluation import check_item_pp_at_end
from src.plotting import plot_pp_vs_x_cat_size
from src.plotting import plot_params
from src.jobs import train_loop
from src.jobs import make_name2seqs
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

TRAIN_DISTANCE = 1
TRAIN_X_CAT_SIZES = [1, 2, 3, 4, 5, 6]
MAX_NUM_EPOCHS = 100
NUM_REPS = 10
PROGRESS_BAR = True
LIMIT_BPPT = True  # if True, generalization to unseen distances is impossible


# params
input_params = config.Input  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt such that it is possible to learn dependencies across largest distance
setattr(rnn_params, 'bptt', config.Eval.max_distance + 1)
print('Set bptt to {}'.format(rnn_params.bptt))

# init result data structures
seq_names = ('train', 'test') if config.Eval.skip_novel else ('train', 'test', 'novel')
distances = np.arange(1, config.Eval.max_distance + 1)
name2dist2item_pps_end = {seq_name: {dist: np.zeros(len(TRAIN_X_CAT_SIZES)) for dist in distances}
                          for seq_name in seq_names}
name2dist2cat_pps_end = {seq_name: {dist: np.zeros(len(TRAIN_X_CAT_SIZES)) for dist in distances}
                         for seq_name in seq_names}

# do for each train x category size
for size_id, train_x_cat_size in enumerate(TRAIN_X_CAT_SIZES):

    # modify input_params before generating sequences
    setattr(input_params, 'train_x_cat_size', train_x_cat_size)
    setattr(input_params, 'min_distance', TRAIN_DISTANCE)
    setattr(input_params, 'max_distance', TRAIN_DISTANCE)

    if LIMIT_BPPT:  # sets bptt to maximal bptt needed to learn training dependencies only
        setattr(rnn_params, 'bptt', TRAIN_DISTANCE + 1)
        print('Set bptt to {}'.format(rnn_params.bptt))

    # make train and test sequences
    train_corpus = AxbCorpus(input_params, test=False)
    test_corpus = AxbCorpus(input_params,  test=True)
    master_vocab = Vocab(train_corpus, test_corpus)
    name2seqs = make_name2seqs(master_vocab, train_corpus, test_corpus)
    seq_names = name2seqs.keys()

    # set max epoch
    setattr(rnn_params, 'num_epochs', MAX_NUM_EPOCHS)

    # progressbar
    print('Training {} models...'.format(NUM_REPS))
    pbar = pyprind.ProgBar(NUM_REPS, stream=sys.stdout)

    # train and evaluate multiple models per hyper-parameter configuration
    for _ in range(NUM_REPS):

        # train + evaluate
        rnn = RNN(master_vocab, rnn_params)
        name2dist2cat_pps, name2dist2item_pps = train_loop(rnn, name2seqs, master_vocab)

        # check item-perplexity against theory
        if not PROGRESS_BAR:
            check_item_pp_at_end(
                rnn, input_params, master_vocab, name2seqs, name2dist2item_pps)

        # populate result data structures
        for seq_name, dist in product(seq_names, distances):
            item_pps = name2dist2item_pps[seq_name][dist]
            cat_pps = name2dist2cat_pps[seq_name][dist]
            if not item_pps or not cat_pps:
                continue
            # item-perplexity
            name2dist2item_pps_end[seq_name][dist][size_id] = item_pps[-1] / NUM_REPS
            # category-perplexity
            name2dist2cat_pps_end[seq_name][dist][size_id] = cat_pps[-1] / NUM_REPS

        if PROGRESS_BAR:
            pbar.update()

# plot perplexity as function of x category size
for name in seq_names:
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    plot_params(time_stamp, input_params, rnn_params)
    plot_pp_vs_x_cat_size(name2dist2item_pps_end[name], TRAIN_X_CAT_SIZES, name, 'Item')