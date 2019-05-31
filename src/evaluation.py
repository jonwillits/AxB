import numpy as np
from scipy.special import softmax

from src import config


def check_type_pp_at_end(srn, input_params, master_vocab, name2seqs, name2dist2type_pps):
    # calculate theoretical maximum and minimum perplexity
    # "B" type perplexity should converge on 1.0, even with variable size distance
    seq_names = name2seqs.keys()
    for seq_name in seq_names:
        distances = np.arange(1, config.Eval.max_distance + 1)
        for dist in distances:
            filtered_seqs = [seq for seq in name2seqs[seq_name] if len(seq) == 3 + dist]
            if not filtered_seqs:
                continue
            x, y = srn.to_x_and_y(filtered_seqs)
            num_windows = len(y)
            num_b_windows = np.sum([1 if master_vocab.types[yi].startswith('B') else 0 for yi in y])
            max_b_type_pp = np.exp(-np.log(1 / input_params.num_ab_types) * num_b_windows / num_windows)
            min_b_type_pp = np.exp(-np.log(1 / 1) * num_b_windows / num_windows)
            #
            b_type_pp_at_start = name2dist2type_pps[seq_name][dist][0]
            b_type_pp_at_end = name2dist2type_pps[seq_name][dist][-1]
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


def calc_pps(srn, master_vocab, name2seqs, name2dist2cat_pps, name2dist2type_pps,):
    seq_names = name2dist2cat_pps.keys()
    for seq_name in seq_names:
        for dist in range(1, config.Eval.max_distance + 1):

            # remove sequences not matching distance
            punctuation = '.' in master_vocab.types
            filtered_seqs = [seq for seq in name2seqs[seq_name] if len(seq) == 2 + int(punctuation) + dist]
            if not filtered_seqs:
                continue

            if config.Verbosity.cat_pp or config.Verbosity.type_pp:
                print('Evaluating at distance={}'.format(dist))
                print('Total number of sequences={} reduced to {}'.format(
                    len(name2seqs[seq_name]), len(filtered_seqs)))

            # logits and softmax probabilities
            x, y = srn.to_x_and_y(filtered_seqs)
            one_hots = np.eye(master_vocab.num_types)[y]
            all_logits = srn.calc_logits(filtered_seqs)
            all_probs = softmax(all_logits, axis=1)

            # TODO make b_probs be in first position always

            # get values for type B only
            num_b = len([t for t in master_vocab.types if t.startswith('B')])
            b_probs = all_probs[:, :num_b]
            b_logits = all_logits[:, :num_b]
            b_onehot =  one_hots[:, :num_b]

            # perplexity for category
            predictions = b_probs.sum(axis=1)  # sum() explains why initial pp differ between categories
            targets = np.array([1 if 'B' in master_vocab.types[yi] else 0 for yi in y])
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

            # perplexity for type
            predictions = softmax(b_logits, axis=1)
            targets = b_onehot
            pp_type = np.exp(calc_cross_entropy(predictions, targets))
            name2dist2type_pps[seq_name][dist].append(pp_type)
            #
            if config.Verbosity.type_pp:
                print(b_logits.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_type)
                print()
    if config.Verbosity.cat_pp or config.Verbosity.type_pp:
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