import numpy as np
import datetime
import sys
import pyprind
from itertools import permutations
from shapely.geometry import Polygon

from src.plotting import plot_pp_trajs
from src.plotting import plot_params
from src.jobs import train_loop
from src.corpus import MarcusCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

"""
note: the pattern 'xxy' and 'xyx' is the only pattern combination where each is learned equally fast.

note: attributes of config.Marcus are class attributes. 
changing these attributes will change them everywhere. they cannot be copied.
"""

PATTERN1_LIST = ['xyy']  # ['xyy', 'xxy', 'xyx']

NUM_EPOCHS_LIST = [0, 1]  # normal training on some pattern
NUM_POST_TRAIN_EPOCHS = 10  # follow-up training on either pattern-consistent or inconsistent test sequences
NUM_REPS = 100
PROGRESS_BAR = True

# perplexity is evaluated at a specific position (pos) in each sequence
# 'knowledge' of a pattern is defined as below-chance by perplexity at position at which repetition occurs
pattern2eval_pos = {'xyy': 2, 'xxy': 1, 'xyx': 2}  # position in sequence which is repeated (0-index)

# params
input_params = config.Marcus  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt
setattr(rnn_params, 'bptt', int(input_params.punctuation) + int(input_params.punctuation_at_start) + 2)
print('Set bptt to {}'.format(rnn_params.bptt))

# for all possible permutations of patterns of size 2
for pattern1, pattern2 in permutations(['xyy', 'xxy', 'xyx'], 2):

    if pattern1 not in PATTERN1_LIST:  # consider only some pattern, if so desired
        continue

    # vary amount of 'normal' (pre-) training
    for num_epochs in NUM_EPOCHS_LIST:

        # init result data structures
        pattern2pps = {pattern1: {'item_pps': np.zeros(NUM_POST_TRAIN_EPOCHS + 1)},  # + 1 for extra eval time point
                       pattern2: {'item_pps': np.zeros(NUM_POST_TRAIN_EPOCHS + 1)}}

        # make corpora + vocab
        train_corpus = MarcusCorpus(input_params, name='train', pattern=pattern1)
        test_corpus1 = MarcusCorpus(input_params, name='test1', pattern=pattern1)
        test_corpus2 = MarcusCorpus(input_params, name='test2', pattern=pattern2)
        master_vocab = Vocab(train_corpus, test_corpus1, test_corpus2)  # create vocab once
        assert train_corpus.params.pattern == test_corpus1.params.pattern
        assert train_corpus.params.pattern != test_corpus2.params.pattern

        # progressbar
        if PROGRESS_BAR:
            print('Training {} models...'.format(NUM_REPS))
        pbar = pyprind.ProgBar(NUM_REPS * 2, stream=sys.stdout)

        for test_corpus in [test_corpus1, test_corpus2]:

            # determine what pp to evaluate
            pattern = test_corpus.params.pattern
            pos = pattern2eval_pos[pattern]  # position in sequence at which item_pp should be below chance
            pattern_cat = pattern[pos]
            cat = {'x': 'C', 'y': 'D'}[pattern_cat]

            # train and evaluate multiple models per hyper-parameter configuration
            for _ in range(NUM_REPS):

                # rnn
                rnn = RNN(master_vocab, rnn_params)

                # 'normal' training
                if not PROGRESS_BAR:
                    print('Normal training with pattern={}...'.format(pattern))
                rnn.params.num_epochs = num_epochs
                train_loop(rnn, master_vocab,
                           train_seqs=master_vocab.make_index_sequences(train_corpus))

                # post-training
                if not PROGRESS_BAR:
                    print('Post-training with pattern={}...'.format(pattern))
                rnn.params.num_epochs = NUM_POST_TRAIN_EPOCHS
                corpus2results = train_loop(rnn, master_vocab,
                                            train_seqs=master_vocab.make_index_sequences(test_corpus))

                # populate result data structures
                if not PROGRESS_BAR:
                    print('Retrieving item_pps computed on cat={} and pos={}'.format(cat, pos))
                assert len(corpus2results[test_corpus.name][cat][pos]['item_pps']) > 0
                item_pps = corpus2results[test_corpus.name][cat][pos]['item_pps']
                pattern2pps[pattern]['item_pps'] += np.asarray(item_pps) / NUM_REPS

                if PROGRESS_BAR:
                    pbar.update()

        # compute area between two perplexity curves
        polygon_points = []
        y1 = list(pattern2pps.values())[0]['item_pps']
        y2 = list(pattern2pps.values())[1]['item_pps']
        x = np.arange(len(y1))
        xy_values1 = list(zip(x, y1))
        xy_values2 = list(zip(x, y2))
        polygon_points.extend(xy_values1)  # append all xy points for curve 1
        polygon_points.extend(xy_values2[::-1])  # append all xy points for curve 2 in the reverse order
        polygon_points.append(xy_values1[0])  # append the first point in curve 1 again, to "close" the polygon
        polygon = Polygon(polygon_points)
        area = polygon.area
        print('polygon area={}'.format(area))

        # plot perplexity
        # note: the two perplexity trajectories are not guaranteed to be computed on the same category and position
        time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
        setattr(input_params, 'pattern', '<see figure>')
        plot_params(time_stamp, input_params, rnn_params)
        plot_pp_trajs(pattern2pps, 'post-training pattern', 'item_pps',
                      x_step=1,
                      y_max=4.0,
                      polygon=polygon,
                      annotation=('Area={:.2f}'.format(area), polygon.centroid.coords[0]),
                      title='Comparing post-training performance after'
                            '\n{}-epoch pre-training with pattern={}'
                            '\nn={}'.format(num_epochs, pattern1, NUM_REPS))