import numpy as np
import datetime
import sys
import pyprind
from itertools import combinations

from src.plotting import plot_pp_trajs
from src.plotting import plot_params
from src.jobs import train_loop
from src.corpus import MarcusCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

PATTERNS = ['xyy', 'xxy', 'xyx']

NUM_EPOCHS = 1
NUM_POST_TRAIN_EPOCHS = 10
NUM_REPS = 10
PROGRESS_BAR = True

pattern2eval_pos = {'xyy': 2, 'xxy': 1, 'xyx': 2}

# params
input_params = config.Marcus  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set RNN params
setattr(rnn_params, 'num_epochs', NUM_EPOCHS)
setattr(rnn_params, 'bptt', int(input_params.punctuation) + int(input_params.punctuation_at_start) + 2)
print('Set bptt to {}'.format(rnn_params.bptt))

for pattern1, pattern2 in combinations(['xyy', 'xxy', 'xyx'], 2):

    if pattern1 not in PATTERNS:
        continue

    # init result data structures
    pattern2pps = {pattern1: {'item_pps': np.zeros(NUM_POST_TRAIN_EPOCHS + 1)},  # + 1 for extra eval time point
                   pattern2: {'item_pps': np.zeros(NUM_POST_TRAIN_EPOCHS + 1)}}

    # make train_corpus
    setattr(input_params, 'pattern', pattern1)
    train_corpus = MarcusCorpus(input_params, name='train')

    # progressbar
    print('Training {} models...'.format(NUM_REPS))
    pbar = pyprind.ProgBar(NUM_REPS * 2, stream=sys.stdout)

    for is_consistent in [True, False]:

        if is_consistent:
            pattern = pattern1
        else:
            pattern = pattern2

        # determine what pp to evaluate
        pos = pattern2eval_pos[pattern]  # position in sequence at which item_pp should be below chance
        pattern_cat = pattern[pos]
        cat = {'x': 'C', 'y': 'D'}[pattern_cat]

        # make test corpus (either consistent or inconsistent with pattern seen during training)
        setattr(input_params, 'pattern', pattern)
        test_corpus = MarcusCorpus(input_params, name='test')

        # train and evaluate multiple models per hyper-parameter configuration
        for _ in range(NUM_REPS):

            # train
            master_vocab = Vocab(train_corpus)
            rnn = RNN(master_vocab, rnn_params)
            train_loop(rnn, master_vocab)

            # post-train on sequences with either pattern1 or pattern2
            master_vocab = Vocab(test_corpus)  # this needs to be done to train on test corpus
            rnn.params.num_epochs = NUM_POST_TRAIN_EPOCHS
            corpus2results = train_loop(rnn, master_vocab)

            # populate result data structures
            assert len(corpus2results['test'][cat][pos]['item_pps']) > 0
            item_pps = corpus2results['test'][cat][pos]['item_pps']
            pattern2pps[pattern]['item_pps'] += np.asarray(item_pps) / NUM_REPS

            if PROGRESS_BAR:
                pbar.update()

    # plot perplexity
    # note: the two perplexity trajectories are not guaranteed to be computed on the same category and position
    time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
    plot_params(time_stamp, input_params, rnn_params)
    plot_pp_trajs(pattern2pps, 'post-training pattern', 'item_pps',
                  x_step=1, title='Comparing post-training performance after'
                                  '\n{}-epoch pre-training with pattern={}'.format(NUM_EPOCHS, pattern1))