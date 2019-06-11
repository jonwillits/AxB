import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from itertools import product

from src import config
from src.utils import to_string


def plot_cat_and_item_pps(name2dist2cat_pps, name2dist2item_pps, seq_names, max_cat_pp):
    for name in seq_names:
        plot_pp_trajs(name2dist2cat_pps[name], name, 'Category', y_max=max_cat_pp)
    for name in seq_names:
        plot_pp_trajs(name2dist2item_pps[name], name, 'Type', y_max=None)


def plot_pp_trajs(dist2pps, title, ylabel_prefix, figsize=(8, 8), fontsize=14, x_step=10, y_max=None, grid=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=None)
    plt.title('sequences = "{}"'.format(title), fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.set_ylabel('{} Perplexity'.format(ylabel_prefix), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if grid:
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
    if y_max is not None:
        ax.set_ylim([1, y_max])
    else:
        all_vals = np.concatenate([pps for pps in dist2pps.values()])
        ax.set_ylim([1, np.max(all_vals)])
    # plot
    x = None
    num_trajs = len(dist2pps)
    palette = iter(sns.color_palette('hls', num_trajs))
    for dist, pps in sorted(dist2pps.items(), key=lambda i: i[0]):
        num_pps = len(pps)
        x = np.arange(0, num_pps + 1, x_step)
        c = next(palette)
        ax.plot(pps, '-', color=c, label='distance={}'.format(dist))
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    #
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()


def plot_pp_vs_x_cat_size(pps_at_end, x, num_reps, ylabel_prefix, figsize=(8, 8), fontsize=14, y_max=None, grid=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=None)
    plt.title('n={}'.format(num_reps), fontsize=fontsize)
    ax.set_xlabel('Size of Category X', fontsize=fontsize)
    ax.set_ylabel('{} Perplexity'.format(ylabel_prefix), fontsize=fontsize)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    if grid:
        ax.yaxis.grid(True)
        ax.xaxis.grid(True)
    if y_max is not None:
        ax.set_ylim([1, y_max])

    # plot
    palette = iter(sns.color_palette('hls', 1))
    c = next(palette)
    ax.plot(x, pps_at_end, '-', color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    #
    plt.tight_layout()
    plt.show()


def plot_grid_search_results_marcus(time_stamp, pp_name, name2cat2pp_mat, name2cat2pp_start, pattern,
                                    seq_names, num_epochs, num_reps, ytick_labels, xtick_labels, ylabel, xlabel,
                                    fontsize=16):
    if len(seq_names) == 2:
        height = 12
    elif len(seq_names) == 3:
        height = 18
    else:
        raise AttributeError('Invalid number of seq_names')
    cats = ['A', 'B']
    num_cats = len(cats)
    # fig
    fig = plt.figure(1, figsize=(26, height))
    gs1 = gridspec.GridSpec(len(seq_names), num_cats)
    axarr = [fig.add_subplot(ss) for ss in gs1]
    for ax, (seq_name, cat) in zip(axarr, product(seq_names, cats)):
        ax.set_title('sequence_name="{}" cat="{}"'.format(seq_name, cat), fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        # heatmap
        mat = name2cat2pp_mat[seq_name][cat]
        if np.count_nonzero(mat) == 0:
            ax.set_axis_off()
        else:
            vmax = name2cat2pp_start[seq_name][cat]
            vmin = 1.0
            im = ax.imshow(mat,
                           aspect='auto',
                           cmap='gray',
                           interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            # label each element
            text_colors = ['black', 'white']
            half_range = (vmax - vmin) / 2
            threshold = im.norm(half_range + vmin)  # threshold below which label is white (instead of black)
            valfmt = ticker.StrMethodFormatter("{x:.4f}")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    is_below = im.norm(mat[i, j]) < threshold
                    color = text_colors[int(is_below)]
                    im.axes.text(j, i, valfmt(mat[i, j], None),
                                 fontsize=fontsize,
                                 horizontalalignment="center",
                                 verticalalignment="center",
                                 color=color)
        # xticks
        num_cols = len(mat.T)
        ax.set_xticks(np.arange(num_cols))
        ax.xaxis.set_ticklabels(xtick_labels, rotation=0, fontsize=fontsize)
        # yticks
        num_rows = len(mat)
        ax.set_yticks(np.arange(num_rows))
        ax.yaxis.set_ticklabels(ytick_labels,  # no need to reverse (because no extent is set)
                                rotation=0, fontsize=fontsize)
        # remove ticklines
        lines = (ax.xaxis.get_ticklines() +
                 ax.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
    #
    fig.suptitle('{}-Perplexity\npattern="{}"\nepoch={}\nn={}\n{}'.format(
        pp_name, pattern, num_epochs, num_reps, time_stamp), fontsize=fontsize)
    gs1.tight_layout(fig, rect=[0, 0, 1, 0.90])
    plt.show()


def plot_grid_search_results(time_stamp, cat, pp_name, name2dist2pp_mat, name2dist2pp_start, seq_names,
                             num_epochs, num_reps, ytick_labels, xtick_labels, ylabel, xlabel, fontsize=16):
    if len(seq_names) == 2:
        height = 12
    elif len(seq_names) == 3:
        height = 18
    else:
        raise AttributeError('Invalid number of seq_names')
    distances = np.arange(1, config.Eval.max_distance + 1)
    # fig
    fig = plt.figure(1, figsize=(26, height))
    gs1 = gridspec.GridSpec(len(seq_names), config.Eval.max_distance)
    axarr = [fig.add_subplot(ss) for ss in gs1]
    for ax, (seq_name, dist) in zip(axarr, product(seq_names, distances)):
        ax.set_title('sequence_name="{}" distance={}'.format(seq_name, dist), fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        # heatmap
        mat = name2dist2pp_mat[seq_name][dist]
        if np.count_nonzero(mat) == 0:
            ax.set_axis_off()
        else:
            vmax = name2dist2pp_start[seq_name][dist]
            vmin = 1.0
            im = ax.imshow(mat,
                           aspect='auto',
                           cmap='gray',
                           interpolation='nearest',
                           vmin=vmin, vmax=vmax)
            # label each element
            text_colors = ['black', 'white']
            half_range = (vmax - vmin) / 2
            threshold = im.norm(half_range + vmin)  # threshold below which label is white (instead of black)
            valfmt = ticker.StrMethodFormatter("{x:.4f}")
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    is_below = im.norm(mat[i, j]) < threshold
                    color = text_colors[int(is_below)]
                    im.axes.text(j, i, valfmt(mat[i, j], None),
                                 fontsize=fontsize,
                                 horizontalalignment="center",
                                 verticalalignment="center",
                                 color=color)
        # xticks
        num_cols = len(mat.T)
        ax.set_xticks(np.arange(num_cols))
        ax.xaxis.set_ticklabels(xtick_labels, rotation=0, fontsize=fontsize)
        # yticks
        num_rows = len(mat)
        ax.set_yticks(np.arange(num_rows))
        ax.yaxis.set_ticklabels(ytick_labels,  # no need to reverse (because no extent is set)
                                rotation=0, fontsize=fontsize)
        # remove ticklines
        lines = (ax.xaxis.get_ticklines() +
                 ax.yaxis.get_ticklines())
        plt.setp(lines, visible=False)
    #
    fig.suptitle('{}-Perplexity for "{}"\nepoch={}\nn={}\n{}'.format(
        pp_name, cat, num_epochs, num_reps, time_stamp), fontsize=fontsize)
    gs1.tight_layout(fig, rect=[0, 0, 1, 0.90])
    plt.show()


def plot_params(time_stamp, input_params, rnn_params, fontsize=14):
    fig = plt.figure(1, figsize=(8, 4), dpi=None)
    # ax1
    ax1 = fig.add_axes([0.1, 0.1, 0.3, 0.0])
    ax1.set_title(to_string(input_params), fontsize=fontsize)
    ax1.set_axis_off()
    # ax2
    ax2 = fig.add_axes([0.6, 0.1, 0.3, 0.0])
    ax2.set_title(to_string(rnn_params), fontsize=fontsize)
    ax2.set_axis_off()
    plt.suptitle('{}'.format(time_stamp), fontsize=fontsize)
    plt.show()
