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


def evaluate(srn, master_vocab, seqs_data, name2cat2pps, name2cat2type_pps, verbose=False):
    for seqs, name in seqs_data:
        if verbose:
            print('Evaluating on {} sequences...'.format(name))
        all_windows = np.vstack([srn.to_windows(seq) for seq in seqs])
        y = all_windows[:, -1]
        onehots = np.eye(master_vocab.master_vocab_size)[y]
        logits = srn.calc_logits(seqs)
        all_probs = softmax(logits, axis=1)
        split_indices = [1, 3, 5, 13]   # TODO make dynamic
        punct_probs, a_probs, b_probs, x_probs = np.split(all_probs, split_indices, axis=1)[:-1]
        punct_onehot, a_onehot, b_onehot, x_onehot = np.split(onehots, split_indices, axis=1)[:-1]

        # perplexity for category
        for probs, stimulus_category in [(punct_probs, '.'),
                                         (a_probs, 'A'),
                                         (b_probs, 'B'),
                                         (x_probs, 'x')]:
            if verbose:
                print('Evaluating using stimulus category="{}"'.format(stimulus_category))
            predictions = probs.sum(axis=1)
            targets = np.array([1 if stimulus_category in master_vocab.master_vocab_list[yi] else 0 for yi in y])
            pp_cat = calc_cross_entropy(predictions, targets)
            #
            if verbose:
                print(probs.round(2))
                print(predictions.round(2))
                print(targets)
                print(targets * np.log(predictions + 1e-9).round(2))
                print(pp_cat)
                print()

            name2cat2pps[name][stimulus_category].append(pp_cat)

        # perplexity for type
        for predictions, targets, stimulus_category in [(punct_probs, punct_onehot, '.'),
                                                  (a_probs, a_onehot, 'A'),
                                                  (b_probs, b_onehot, 'B'),
                                                  (x_probs, x_onehot, 'x')]:
            pp_type = calc_cross_entropy(predictions, targets)
            name2cat2type_pps[name][stimulus_category].append(pp_type)

        if verbose:
            print('------------------------------------------------------------')


def plot_pp_trajs(cat2pps, title, ylabel_prefix, figsize=(8, 8), fontsize=14, x_step=10):
    fig, ax = plt.subplots(figsize=figsize, dpi=None)
    plt.title('sequences = "{}"'.format(title), fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('{} Perplexity'.format(ylabel_prefix), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.yaxis.grid(True)
    # ax.xaxis.grid(True)
    ax.set_ylim([0, 1.0])
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