import numpy as np
from scipy.special import softmax

from src import config


def calc_pps(srn, master_vocab, seqs, seq_name, name2cat2pps, name2cat2type_pps, split_indices):
    # logits and softmax probabilities
    x, y = srn.to_x_and_y(seqs)
    onehots = np.eye(master_vocab.master_vocab_size)[y]
    all_logits = srn.calc_logits(seqs)
    all_probs = softmax(all_logits, axis=1)
    # split by category
    punct_probs, a_probs, b_probs, x_probs = np.split(all_probs, split_indices, axis=1)[:-1]
    punct_logits, a_logits, b_logits, x_logits = np.split(all_logits, split_indices, axis=1)[:-1]
    punct_onehot, a_onehot, b_onehot, x_onehot = np.split(onehots, split_indices, axis=1)[:-1]

    # perplexity for category
    predictions = b_probs.sum(axis=1)  # sum() explains why initial pp differ between categories
    targets = np.array([1 if 'B' in master_vocab.master_vocab_list[yi] else 0 for yi in y])
    pp_cat = calc_cross_entropy(predictions, targets)
    name2cat2pps[seq_name]['B'].append(pp_cat)
    #
    if config.General.cat_pp_verbose:
        print('Evaluating using stimulus category="{}"'.format('B'))
        print(b_probs.round(2))
        print(predictions.round(2))
        print(targets)
        print(targets * np.log(predictions + 1e-9).round(2))
        print(pp_cat)
        print()

    # perplexity for type
    predictions = softmax(b_logits, axis=1)
    targets = b_onehot
    pp_type = calc_cross_entropy(predictions, targets)
    name2cat2type_pps[seq_name]['B'].append(pp_type)
    #
    if config.General.type_pp_verbose and seq_name == 'train':
        print('Evaluating using stimulus category="{}"'.format('B'))
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