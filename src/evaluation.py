import numpy as np
from scipy.special import softmax

from src import config


def make_name2dist2type_pp_at_end(srn, input_params, master_vocab, name2seqs, name2dist2type_pps):
    # calculate theoretical maximum and minimum perplexity
    # "B" type perplexity should converge on 1.0, even with variable size distance
    seq_names = name2seqs.keys()
    res = {name: {} for name in seq_names}
    for seq_name in seq_names:
        distances = np.arange(input_params.min_distance, input_params.max_distance + 1)
        for dist in distances:
            filtered_seqs = [seq for seq in name2seqs[seq_name] if len(seq) == 3 + dist]
            x, y = srn.to_x_and_y(filtered_seqs)
            num_windows = len(y)
            num_b_windows = np.sum([1 if master_vocab.types[yi].startswith('B') else 0 for yi in y])
            max_b_type_pp = np.exp(-np.log(1 / input_params.num_ab_types) * num_b_windows / num_windows)
            min_b_type_pp = np.exp(-np.log(1 / 1) * num_b_windows / num_windows)
            #
            b_type_pp_at_start = name2dist2type_pps[seq_name][dist][0]
            b_type_pp_at_end = name2dist2type_pps[seq_name][dist][-1]
            res[seq_name][dist] = b_type_pp_at_end
            # console
            print('-------------')
            print('distance={} seq_name={}'.format(dist, seq_name))
            print('-------------')
            print('num_b_windows', num_b_windows)
            print('num_windows', num_windows)
            print('max_b_type_pp', max_b_type_pp)
            print('min_b_type_pp', min_b_type_pp)
            print('b_type_pp_at_start={}'.format(b_type_pp_at_start))
            print('b_type_pp_at_end  ={}'.format(b_type_pp_at_end))
    return res


def calc_pps(srn, master_vocab, name2seqs, distances, name2dist2cat_pps, name2dist2type_pps, split_indices):

    assert '.' in master_vocab.types  # else distance will not be correct
    seq_names = name2dist2cat_pps.keys()
    for seq_name in seq_names:
        for dist in distances:
            filtered_seqs = [seq for seq in name2seqs[seq_name] if len(seq) == 3 + dist]
            if config.General.cat_pp_verbose or config.General.type_pp_verbose:
                print('Evaluating at distance={}'.format(dist))
                print('Total number of sequences={} reduced to {}'.format(
                    len(name2seqs[seq_name]), len(filtered_seqs)))

            # logits and softmax probabilities
            x, y = srn.to_x_and_y(filtered_seqs)
            onehots = np.eye(master_vocab.master_vocab_size)[y]
            all_logits = srn.calc_logits(filtered_seqs)
            all_probs = softmax(all_logits, axis=1)
            # split by category
            punct_probs, a_probs, b_probs, x_probs = np.split(all_probs, split_indices, axis=1)[:-1]
            punct_logits, a_logits, b_logits, x_logits = np.split(all_logits, split_indices, axis=1)[:-1]
            punct_onehot, a_onehot, b_onehot, x_onehot = np.split(onehots, split_indices, axis=1)[:-1]

            # perplexity for category
            predictions = b_probs.sum(axis=1)  # sum() explains why initial pp differ between categories
            targets = np.array([1 if 'B' in master_vocab.types[yi] else 0 for yi in y])
            pp_cat = np.exp(calc_cross_entropy(predictions, targets))
            name2dist2cat_pps[seq_name][dist].append(pp_cat)
            #
            if config.General.cat_pp_verbose:
                print(b_probs.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_cat)
                print()

            # perplexity for type
            predictions = softmax(b_logits, axis=1)
            targets = b_onehot
            pp_type = np.exp(calc_cross_entropy(predictions, targets))
            name2dist2type_pps[seq_name][dist].append(pp_type)
            #
            if config.General.type_pp_verbose:
                # print(b_logits.round(2))
                # print(predictions.round(2))
                # print(targets)
                # print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_type)
                print()
    if config.General.cat_pp_verbose or config.General.type_pp_verbose:
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