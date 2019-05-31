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
            max_b_item_pp = np.exp(-np.log(1 / input_params.num_ab_types) * num_b_windows / num_windows)
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


def calc_pps(srn, master_vocab, name2seqs, name2dist2cat_pps, name2dist2item_pps,):
    for seq_name, seqs in name2seqs.items():
        for dist in range(1, config.Eval.max_distance + 1):

            # remove sequences not matching distance
            punctuation = '.' in master_vocab.items
            filtered_seqs = [seq for seq in seqs if len(seq) == 2 + dist + int(punctuation)]
            if not filtered_seqs:
                continue

            if config.Verbosity.cat_pp or config.Verbosity.item_pp:
                print('Evaluating at distance={}'.format(dist))
                print('Total number of sequences={} reduced to {}'.format(
                    len(seqs), len(filtered_seqs)))

            # logits and softmax probabilities
            x, y = srn.to_x_and_y(filtered_seqs)
            one_hots = np.eye(master_vocab.num_types)[y]
            all_logits = srn.calc_logits(filtered_seqs)
            all_probs = softmax(all_logits, axis=1)

            # get values for category B only
            num_b = len([item for item in master_vocab.items if item.startswith('B')])
            b_probs = all_probs[:, :num_b]
            b_logits = all_logits[:, :num_b]
            b_onehot = one_hots[:, :num_b]

            # perplexity for category
            predictions = b_probs.sum(axis=1)  # sum() explains why initial pp differ between categories
            targets = np.array([1 if master_vocab.items[yi].startswith('B') else 0 for yi in y])
            pp_cat = np.exp(calc_cross_entropy(predictions, targets))
            name2dist2cat_pps[seq_name][dist].append(pp_cat)
            #
            if config.Verbosity.cat_pp:
                print(b_probs.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_cat)
                print()

            # perplexity for item
            predictions = softmax(b_logits, axis=1)
            targets = b_onehot
            pp_type = np.exp(calc_cross_entropy(predictions, targets))
            name2dist2item_pps[seq_name][dist].append(pp_type)
            #
            if config.Verbosity.item_pp:
                print(b_logits.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_type)
                print()

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