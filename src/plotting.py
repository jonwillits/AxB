import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec


def plot_cat_and_type_pps(name2dist2cat_pps, name2dist2type_pps, seq_names, max_cat_pp):
    for name in seq_names:
        plot_pp_trajs(name2dist2cat_pps[name], name, 'Category', y_max=max_cat_pp)
    for name in seq_names:
        plot_pp_trajs(name2dist2type_pps[name], name, 'Type', y_max=None)


def plot_pp_trajs(dist2pps, title, ylabel_prefix, figsize=(8, 8), fontsize=14, x_step=10, y_max=None):
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
        ax.set_ylim([1, y_max])
    else:
        all_vals = np.concatenate([pps for pps in dist2pps.values()])
        ax.set_ylim([1, np.max(all_vals)])
    # plot
    xticks = None
    num_trajs = len(dist2pps)
    palette = iter(sns.color_palette('hls', num_trajs))
    for dist, pps in sorted(dist2pps.items(), key=lambda i: i[0]):
        num_pps = len(pps)
        xticks = np.arange(0, num_pps + 1, x_step)
        c = next(palette)
        ax.plot(pps, '-', color=c, label='distance={}'.format(dist))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    #
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    plt.show()


def plot_grid_mat(dist2grid_mat, max_num_epochs, ytick_labels, xtick_labels, ylabel, xlabel,
                  figsize=(16, 8), dpi=None, fontsize=14):
    distances = dist2grid_mat.keys()
    num_distances = len(distances)
    fig = plt.figure(1)
    gs1 = gridspec.GridSpec(1, num_distances)
    axarr = [fig.add_subplot(ss) for ss in gs1]
    for ax, dist in zip(axarr, distances):
        ax.set_title('distance={}'.format(dist), fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        # heatmap
        print('Plotting heatmap...')
        mat = dist2grid_mat[dist]
        im = ax.imshow(mat,
                       aspect='auto',
                       cmap='gray',
                       interpolation='nearest')
        # label each element
        text_colors = ['black', 'white']
        threshold = im.norm(np.max(mat) / 1)  # threshold below which label is white (instead of black)
        valfmt = ticker.StrMethodFormatter("{x:.4f}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                is_below = im.norm(mat[i, j]) < threshold
                color = text_colors[int(is_below)]
                im.axes.text(j, i, valfmt(mat[i, j], None),
                             fontsize=fontsize+2,
                             horizontalalignment="center",
                             verticalalignment="center",
                             color=color)
        # xticks
        num_cols = len(mat.T)
        ax.set_xticks(np.arange(num_cols))
        ax.xaxis.set_ticklabels(xtick_labels, rotation=90, fontsize=fontsize)
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
    fig.suptitle('Type Perplexity for "B"\nepoch={}'.format(max_num_epochs), fontsize=fontsize)
    gs1.tight_layout(fig, rect=[0, 0, 1, 0.90])
    plt.show()