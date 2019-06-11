import numpy as np
from scipy.special import softmax

from src import config


def check_item_pp_at_end(srn, input_params, master_vocab, name2seqs, name2dist2item_pps):
    # calculate theoretical maximum and minimum perplexity
    # "B" item perplexity should converge on 1.0, even with variable size distance
    for seq_name, seqs in name2seqs.items():
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
            b_item_pp_at_start = name2dist2item_pps[seq_name][dist][0]
            b_item_pp_at_end = name2dist2item_pps[seq_name][dist][-1]

            # console
            print('-------------')
            print('distance={} seq_name={}'.format(dist, seq_name))
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


def update_cat_and_item_pps(srn, master_vocab, name2seqs, cat, distances, results):  # TODO adapt to marcus
    for seq_name, seqs in name2seqs.items():
        if config.Verbosity.cat_pp or config.Verbosity.item_pp:
            print('Evaluating perplexity for "{}" on "{}" sequences'.format(cat, seq_name))

        # logits and softmax probabilities - calculate once only, and then filter by distance
        x, all_y = srn.to_x_and_y(seqs)
        all_onehots = np.eye(master_vocab.num_items)[all_y]
        all_logits = srn.calc_logits(seqs)
        all_probs = softmax(all_logits, axis=1)

        if distances is None:  # Marcus corpus
            # filter by category
            is_cat_bools = [True if item.startswith(cat) else False for item in master_vocab.items]
            filtered_probs = all_probs[:, is_cat_bools]
            filtered_logits = all_logits[:, is_cat_bools]
            filtered_onehots = all_onehots[:, is_cat_bools]
            filtered_y = all_y
            # compute
            cat_pp = calc_cat_pp(master_vocab, cat, filtered_probs, filtered_y)
            item_pp = calc_item_pp(filtered_logits, filtered_onehots)
            # collect
            results[0][seq_name].append(cat_pp)
            results[1][seq_name].append(item_pp)

        else:  # AxB corpus
            for dist in distances:
                # filter by distance & category
                is_dist_bools = make_is_dist_bools(srn, master_vocab, seqs, dist)
                if not np.any(is_dist_bools):
                    print('Did not find sequences with distance={}. Skipping'.format(dist))
                    continue
                is_cat_bools = [True if item.startswith(cat) else False for item in master_vocab.items]
                filtered_probs = all_probs[is_dist_bools][:, is_cat_bools]
                filtered_logits = all_logits[is_dist_bools][:, is_cat_bools]
                filtered_onehots = all_onehots[is_dist_bools][:, is_cat_bools]
                filtered_y = all_y[is_dist_bools]
                # compute
                cat_pp = calc_cat_pp(master_vocab, cat, filtered_probs, filtered_y)
                item_pp = calc_item_pp(filtered_logits, filtered_onehots)
                # collect
                results[0][seq_name][dist].append(cat_pp)
                results[1][seq_name][dist].append(item_pp)

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