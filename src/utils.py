import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns


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


def evaluate(srn, master_vocab, seqs_data, name2cat2pps, name2cat2type_pps, verbose_cat, verbose_type):
    for seqs, name in seqs_data:
        if verbose_cat or verbose_type:
            print('Evaluating on {} sequences...'.format(name))
        all_windows = np.vstack([srn.to_windows(seq) for seq in seqs])
        y = all_windows[:, -1]
        onehots = np.eye(master_vocab.master_vocab_size)[y]
        all_logits = srn.calc_logits(seqs)
        all_probs = softmax(all_logits, axis=1)
        split_indices = [1, 3, 5, 13]   # TODO make dynamic
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
            if verbose_cat:
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
            if verbose_type:
                print('Evaluating using stimulus category="{}"'.format(stimulus_category))
                print(logits.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_type)
                print()

        if verbose_cat or verbose_type:
            print('------------------------------------------------------------')


def plot_pp_trajs(cat2pps, title, ylabel_prefix, figsize=(8, 8), fontsize=14, x_step=10, y_max=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=None)
    plt.title('sequences = "{}"'.format(title), fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('{} Perplexity'.format(ylabel_prefix), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    if y_max is not None:
        ax.set_ylim([0, y_max])
    # plot
    xticks = None
    num_trajs = len(cat2pps)
    palette = iter(sns.color_palette('hls', num_trajs))
    for cat, pps in sorted(cat2pps.items(), key=lambda i: i[0]):
        num_pps = len(pps)
        xticks = np.arange(0, num_pps + 1, x_step)
        c = next(palette)
        ax.plot(pps, '-', color=c, label='stimulus category="{}"'.format(cat))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    #
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()