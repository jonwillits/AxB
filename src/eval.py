import numpy as np
from scipy.special import softmax

from src import config


def calc_accuracies(srn, seqs, master_vocab):
    x, y = srn.to_x_and_y(seqs)
    logits = srn.calc_logits(seqs)
    # init counts
    correct_count = [0, 0, 0, 0]
    n = [0, 0, 0, 0]
    acc = [0, 0, 0, 0]
    #
    for i in range(len(y)):
        correct_index = y[i]
        guess_index = np.argmax(logits[i])
        correct_label = master_vocab.master_vocab_list[correct_index]
        guess_label = master_vocab.master_vocab_list[guess_index]
        #
        if correct_index == guess_index:
            correct = 1
        else:
            correct = 0
        if correct_label.startswith('A'):
            category = 0
        elif correct_label.startswith('x'):
            category = 1
        elif correct_label.startswith('B'):
            category = 2
        elif correct_label.startswith('.'):
            category = 3
        else:
            raise AttributeError('Invalid arg to "category"')
        correct_count[category] += correct
        n[category] += 1
        for j in range(len(n)):
            if n[j] == 0:
                acc[j] = -1
            else:
                acc[j] = float(correct_count[j]) / n[j]
    return acc


def calc_pps(srn, master_vocab, seqs_data, name2cat2pps, name2cat2type_pps, split_indices):
    for seqs, name in zip(*seqs_data):
        if config.General.cat_pp_verbose or config.General.type_pp_verbose:
            print('Evaluating on {} sequences...'.format(name))
        # logits and softmax probabilities
        all_windows = np.vstack([srn.to_windows(seq) for seq in seqs])
        y = all_windows[:, -1]
        onehots = np.eye(master_vocab.master_vocab_size)[y]
        all_logits = srn.calc_logits(seqs)
        all_probs = softmax(all_logits, axis=1)
        # split by category
        punct_probs, a_probs, b_probs, x_probs = np.split(all_probs, split_indices, axis=1)[:-1]
        punct_logits, a_logits, b_logits, x_logits = np.split(all_logits, split_indices, axis=1)[:-1]
        punct_onehot, a_onehot, b_onehot, x_onehot = np.split(onehots, split_indices, axis=1)[:-1]

        # perplexity for category
        for probs, stimulus_category in [(punct_probs, '.'),
                                         (a_probs, 'A'),
                                         (b_probs, 'B'),
                                         (x_probs, 'x')]:
            predictions = probs.sum(axis=1)  # sum() explains why initial pp differ between categories
            targets = np.array([1 if stimulus_category in master_vocab.master_vocab_list[yi] else 0 for yi in y])
            pp_cat = calc_cross_entropy(predictions, targets)
            name2cat2pps[name][stimulus_category].append(pp_cat)
            #
            if config.General.cat_pp_verbose:
                print('Evaluating using stimulus category="{}"'.format(stimulus_category))
                print(probs.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_cat)
                print()

        # perplexity for type
        for logits, targets, stimulus_category in [(punct_logits, punct_onehot, '.'),
                                                   (a_logits, a_onehot, 'A'),
                                                   (b_logits, b_onehot, 'B'),
                                                   (x_logits, x_onehot, 'x')]:
            if logits.shape[1] == 1:  # it doesn't make sense to compute softmax when there are no choice ("."
                continue
            predictions = softmax(logits, axis=1)
            pp_type = calc_cross_entropy(predictions, targets)
            name2cat2type_pps[name][stimulus_category].append(pp_type)
            #
            if config.General.type_pp_verbose:
                print('Evaluating using stimulus category="{}"'.format(stimulus_category))
                print(logits.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
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