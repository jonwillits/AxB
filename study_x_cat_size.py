import numpy as np
import datetime
import sys
import pyprind

from src.evaluation import check_item_pp_at_end
from src.plotting import plot_pp_vs_x_cat_size
from src.plotting import plot_params
from src.jobs import train_loop
from src.utils import calc_max_item_pp
from src.corpus import AxbCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

TRAIN_DISTANCE = 1
TRAIN_X_CAT_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
MAX_NUM_EPOCHS = 2
NUM_REPS = 10
PROGRESS_BAR = True
LIMIT_BPPT = True  # if True, generalization to unseen distances is impossible
MAX_NUM_SEQUENCES = 48  # total number of sequences in corpus with x_cat_size=24


# params
input_params = config.Input  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt such that it is possible to learn dependencies across largest distance
setattr(rnn_params, 'bptt', config.Eval.max_distance + 1)
print('Set bptt to {}'.format(rnn_params.bptt))

# init result data structures
item_pps_end = np.zeros(len(TRAIN_X_CAT_SIZES))
cat_pps_end = np.zeros(len(TRAIN_X_CAT_SIZES))

# do for each train x category size
for size_id, train_x_cat_size in enumerate(TRAIN_X_CAT_SIZES):

    # modify input_params before generating sequences
    setattr(input_params, 'train_x_cat_size', train_x_cat_size)
    setattr(input_params, 'test_x_cat_size', train_x_cat_size)
    setattr(input_params, 'min_distance', TRAIN_DISTANCE)
    setattr(input_params, 'max_distance', TRAIN_DISTANCE)
    setattr(input_params, 'sample_size', MAX_NUM_SEQUENCES)

    max_item_pp = calc_max_item_pp(input_params, 'B')
    print('max_item_pp={}'.format(max_item_pp))

    # make train but not test sequences
    train_corpus = AxbCorpus(input_params, test=False)
    master_vocab = Vocab(train_corpus)
    name2seqs = {'train': master_vocab.generate_index_sequences(train_corpus)}
    seq_names = name2seqs.keys()
    print('number of sequences in train corpus={}'.format(train_corpus.num_sequences))

    # set max epoch
    setattr(rnn_params, 'num_epochs', MAX_NUM_EPOCHS)

    if LIMIT_BPPT:  # sets bptt to maximal bptt needed to learn training dependencies only
        setattr(rnn_params, 'bptt', TRAIN_DISTANCE + 1)
        print('Set bptt to {}'.format(rnn_params.bptt))

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
        dist = TRAIN_DISTANCE
        item_pps = name2dist2item_pps['train'][dist]
        cat_pps = name2dist2cat_pps['train'][dist]
        if not item_pps or not cat_pps:
            continue
        item_pps_end[size_id] += item_pps[-1] / NUM_REPS
        cat_pps_end[size_id] += cat_pps[-1] / NUM_REPS

        if PROGRESS_BAR:
            pbar.update()

# plot perplexity as function of x category size
time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
plot_params(time_stamp, input_params, rnn_params)
plot_pp_vs_x_cat_size(item_pps_end, TRAIN_X_CAT_SIZES, 'train', 'Item', y_max=max_item_pp)


    # TODO keep output size of model constant