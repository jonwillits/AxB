import numpy as np
import datetime
import sys
import pyprind

from src.utils import print_params
from src.utils import make_random_sequences
from src.utils import calc_max_item_pp_marcus
from src.plotting import plot_grid_search_results_marcus
from src.plotting import plot_params
from src.jobs import train_loop
from src.corpus import MarcusCorpus
from src.vocab import Vocab
from src.rnn import RNN
from src import config

PATTERNS = ['xxy'] or ['xyy', 'xxy', 'xyx']
PARAMS1_NAME = 'learning_rate'
PARAMS1 = [0.1, 0.25, 0.5, 0.75, 1.0]
PARAMS2_NAME = 'hidden_size'
PARAMS2 = [2, 4, 6, 8]
NUM_PRETRAIN_EPOCHS = 0  # set to 0 to not pre-train
NUM_EPOCHS = 20
PLOT_SEQ_NAMES = ['train', 'test']  # test = novel
NUM_REPS = 10
PROGRESS_BAR = True

# params
input_params = config.Marcus  # cannot be copied
rnn_params = config.RNN  # cannot be copied

# set bptt
setattr(rnn_params, 'bptt', int(input_params.punctuation) + int(input_params.punctuation_at_start) + 2)
print('Set bptt to {}'.format(rnn_params.bptt))

# do for each marcus corpus pattern
for pattern in PATTERNS:

    # progressbar
    print('Grid search with pattern={}'.format(pattern))
    pbar = pyprind.ProgBar(len(PARAMS1) * len(PARAMS2), stream=sys.stdout)

    # modify input_params before generating sequences
    setattr(input_params, 'pattern', pattern)

    # make train and test corpora
    train_corpus = MarcusCorpus(input_params, name='train')
    test_corpus = MarcusCorpus(input_params,  name='test')
    assert set(train_corpus.cats).isdisjoint(test_corpus.cats)
    master_vocab = Vocab(train_corpus, test_corpus)

    # init result data structures
    name2pos2item_pp_mat = {corpus.name: {pos: np.zeros((len(PARAMS1), len(PARAMS2))) for pos in corpus.positions}
                            for corpus in master_vocab.corpora}
    name2pos2cat_pp_mat = {corpus.name: {pos: np.zeros((len(PARAMS1), len(PARAMS2))) for pos in corpus.positions}
                           for corpus in master_vocab.corpora}
    name2pos2item_pp_start = {corpus.name: {pos: None for pos in corpus.positions}
                              for corpus in master_vocab.corpora}
    name2pos2cat_pp_start = {corpus.name: {pos: None for pos in corpus.positions}
                             for corpus in master_vocab.corpora}

    # grid search over rnn_params
    for i, param1 in enumerate(PARAMS1):
        for j, param2 in enumerate(PARAMS2):

            # overwrite rnn_params
            setattr(rnn_params, PARAMS1_NAME, param1)
            setattr(rnn_params, PARAMS2_NAME, param2)
            setattr(rnn_params, 'num_epochs', NUM_EPOCHS)

            if not PROGRESS_BAR:
                print_params(rnn_params)

            # train and evaluate multiple models per hyper-parameter configuration
            for _ in range(NUM_REPS):

                # train + evaluate
                rnn = RNN(master_vocab, rnn_params)
                items_in_random_seqs = [i for i in range(master_vocab.num_items) if master_vocab.items[i] != '.']
                random_seqs = make_random_sequences(items_in_random_seqs,
                                                    exclude_pattern=pattern,
                                                    seq_size=3,
                                                    num_sequences=len(master_vocab.train_seqs)) * NUM_PRETRAIN_EPOCHS
                corpus2results = train_loop(rnn, master_vocab, pretrain_seqs=random_seqs)

                # populate result data structures
                for corpus in master_vocab.corpora:
                    for cat in corpus.cats:
                        for pos in corpus.positions:
                            cat_pps = corpus2results[corpus.name][cat][pos]['cat_pps']
                            item_pps = corpus2results[corpus.name][cat][pos]['item_pps']
                            if not item_pps or not cat_pps:
                                continue
                            # category-perplexity
                            name2pos2cat_pp_mat[corpus.name][pos][i, j] += cat_pps[-1] / NUM_REPS
                            name2pos2cat_pp_start[corpus.name][pos] = cat_pps[0]
                            # item-perplexity
                            name2pos2item_pp_mat[corpus.name][pos][i, j] += item_pps[-1] / NUM_REPS
                            name2pos2item_pp_start[corpus.name][pos] = item_pps[0]

            if PROGRESS_BAR:
                pbar.update()

    if not (config.Verbosity.cat_pp or config.Verbosity.item_pp):
        # plot heatmaps showing item or category perplexity for all hyper-parameter configurations
        time_stamp = datetime.datetime.now().strftime("%B %d %Y %I:%M:%s")
        setattr(rnn_params, PARAMS1_NAME, '<grid_search>')
        setattr(rnn_params, PARAMS2_NAME, '<grid_search>')
        # plot_params(time_stamp, input_params, rnn_params)
        # plot_grid_search_results_marcus(time_stamp, 'Category', name2pos2cat_pp_mat, name2pos2cat_pp_start, pattern,
        #                                 PLOT_SEQ_NAMES, NUM_EPOCHS, NUM_REPS,
        #                                 PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)
        plot_grid_search_results_marcus(time_stamp, 'Item', name2pos2item_pp_mat, name2pos2item_pp_start, pattern,
                                        PLOT_SEQ_NAMES, NUM_EPOCHS, NUM_REPS,
                                        PARAMS1, PARAMS2, PARAMS1_NAME, PARAMS2_NAME)



