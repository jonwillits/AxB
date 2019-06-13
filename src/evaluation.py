import numpy as np
from scipy.special import softmax

from src import config


def check_b_item_pp_at_end(srn, input_params, master_vocab, corpus2results):
    # calculate theoretical maximum and minimum perplexity
    # "B" item perplexity should converge on 1.0, even with variable size distance (for AxB corpus)
    for corpus in master_vocab.corpora:
        seqs = master_vocab.generate_index_sequences(corpus)
        distances = np.arange(1, config.Eval.max_distance + 1)
        for dist in distances:

            # remove sequences not matching distance
            punctuation = '.' in master_vocab.items
            filtered_seqs = [seq for seq in seqs if len(seq) == 2 + dist + int(punctuation)]
            if not filtered_seqs:
                continue

            x, y = srn.to_x_and_y(filtered_seqs)
            num_windows = len(y)
            num_b_windows = np.sum([1 if master_vocab.items[yi].startswith('B') else 0 for yi in y])
            max_b_item_pp = np.exp(-np.log(1 / input_params.ab_cat_size) * num_b_windows / num_windows)
            min_b_item_pp = np.exp(-np.log(1 / 1) * num_b_windows / num_windows)

            #
            b_item_pp_at_start = name2dist2item_pps[corpus.name][dist][0]
            b_item_pp_at_end = name2dist2item_pps[corpus.name][dist][-1]

            # console
            print('-------------')
            print('distance={} corpus.name={}'.format(dist, corpus.name))
            print('-------------')
            print('num_b_windows', num_b_windows)
            print('num_windows', num_windows)
            print('max_b_item_pp', max_b_item_pp)
            print('min_b_item_pp', min_b_item_pp)
            print('b_item_pp_at_start={}'.format(b_item_pp_at_start))
            print('b_item_pp_at_end  ={}'.format(b_item_pp_at_end))


def make_is_dist_bools(srn, master_vocab, seqs, dist):
    """
    return boolean index into windows. True only if window is part of sequence with allowed distance
    """
    res = []
    for seq in seqs:
        punctuation = '.' in master_vocab.items
        is_distance = len(seq) == 2 + dist + int(punctuation)
        window_bools = [is_distance] * len(srn.to_windows(seq))
        res.extend(window_bools)
    #
    if config.Verbosity.cat_pp or config.Verbosity.item_pp:
        num_filtered_seqs = np.count_nonzero(res)
        print('Total number of sequences={} reduced to {}'.format(len(seqs), num_filtered_seqs))
    return res


def calc_cat_pp(master_vocab, cat, filtered_probs, filtered_y):
    # perplexity for category
    predictions = filtered_probs.sum(axis=1)  # sum() explains why initial pp differ between categories
    targets = np.array([1 if master_vocab.items[yi].startswith(cat) else 0 for yi in filtered_y])
    cat_pp = np.exp(calc_cross_entropy(predictions, targets))
    #
    if config.Verbosity.cat_pp:
        print(filtered_probs.round(2))
        print(predictions.round(2))
        print(targets)
        print(targets * np.log(predictions + 1e-9).round(2))
        print(cat_pp)
        print()
    return cat_pp


def calc_item_pp(filtered_logits, filtered_onehots):
    # perplexity for item
    predictions = softmax(filtered_logits, axis=1)
    targets = filtered_onehots
    item_pp = np.exp(calc_cross_entropy(predictions, targets))
    #
    if config.Verbosity.item_pp:
        print(filtered_logits.round(2))
        print(predictions.round(2))
        print(targets)
        print(targets * np.log(predictions + 1e-9).round(2))
        print(item_pp)
        print()
    return item_pp


def update_cat_and_item_pps(srn, master_vocab, corpus2results):
    for corpus in master_vocab.corpora:

        seqs = master_vocab.generate_index_sequences(corpus)

        for cat in corpus.cats:

            # logits and softmax probabilities - calculate once only, and then filter by distance
            all_x, all_y = srn.to_x_and_y(seqs)
            all_onehots = np.eye(master_vocab.num_items)[all_y]
            all_logits = srn.calc_logits(seqs)
            all_probs = softmax(all_logits, axis=1)

            # filter by columns by category
            is_cat_col_bools = [True if item.startswith(cat) else False for item in master_vocab.items]
            cat_probs = all_probs[:, is_cat_col_bools]
            cat_logits = all_logits[:, is_cat_col_bools]
            cat_onehots = all_onehots[:, is_cat_col_bools]

            # pos_y is a list with integers representing the position of the item that is predicted
            pos_seqs = [list(range(len(seq))) for seq in seqs]
            pos_x, pos_y = srn.to_x_and_y(pos_seqs)
            for pos in corpus.positions:

                # filter rows by position + category
                is_pos_row_bools = [True if pos_yi == pos else False for pos_yi in pos_y]
                is_cat_row_bools = [True if master_vocab.items[yi].startswith(cat) else False for yi in all_y]
                is_pos_cat_bools = np.logical_and(is_pos_row_bools, is_cat_row_bools)
                if not np.any(is_pos_cat_bools):
                    continue  # only evaluate perplexity when cat actually occurs in given position
                probs = cat_probs[is_pos_cat_bools]
                logits = cat_logits[is_pos_cat_bools]
                onehots = cat_onehots[is_pos_cat_bools]
                y = all_y[is_pos_cat_bools]



                if config.Verbosity.cat_pp or config.Verbosity.item_pp:
                    print('Evaluating perplexity:')
                    print('corpus_name="{}"'.format(corpus.name))
                    print('cat="{}"'.format(cat))
                    print('pos={}'.format(pos))

                # compute
                cat_pp = calc_cat_pp(master_vocab, cat, probs, y)
                item_pp = calc_item_pp(logits, onehots)
                # collect
                corpus2results[corpus.name][cat][pos]['cat_pps'].append(cat_pp)
                corpus2results[corpus.name][cat][pos]['item_pps'].append(item_pp)

                if config.Verbosity.cat_pp or config.Verbosity.item_pp:
                    print('------------------------------------------------------------')


def calc_cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce