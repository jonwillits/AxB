import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_cat_and_type_pps(name2cat2cat_pps, name2cat2type_pps, seq_names, max_cat_pp):
    for name in seq_names:
        plot_pp_trajs(name2cat2cat_pps[name], name, 'Category', y_max=max_cat_pp)
    for name in seq_names:
        plot_pp_trajs(name2cat2type_pps[name], name, 'Type', y_max=1.0)


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