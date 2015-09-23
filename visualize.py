import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import StringIO
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn_pandas import DataFrameMapper


sns.set_context("paper")
sns.set_style("white")


def output_figure(filename):
    out = None

    if filename is None:
        plt.show()
    elif filename == '__qcml_export__':
        binary = StringIO.StringIO()
        plt.savefig(binary, format='svg')
        binary.seek(0)  # rewind the data
        out = binary.buf
    else:
        plt.savefig(filename)

    plt.close()

    return out


# Remember to change the matplotlib backend for the heatmap annotation to work!
def visualize_correlation_matrix(corr, filename=None):
    plt.figure(figsize=(11, 10))

    # generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, vmin=-1, vmax=1, linewidths=.5, square=True,
                xticklabels=corr.columns.values[:-1], yticklabels=corr.columns.values[1:],
                mask=mask, cbar_kws={'shrink': .75}, annot=True, fmt='.2f', annot_kws={"size": 4})

    # rotate overly long tick labels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return output_figure(filename)


def classes_to_colors(df):
    cmap = plt.cm.get_cmap('autumn')(np.linspace(0, 1, len(df.index.levels[0])))

    class_colors = {}
    color_idx = 0
    for c, _ in df.index.values:
        if class_colors.get(c) is None:
            class_colors[c] = cmap[color_idx]
            color_idx += 1

    colors = []
    for c, _ in df.index.values:
        colors.append(class_colors[c])

    return colors


def visualize_timestamps(df, filename=None):
    plt.figure()

    plt.scatter(df.index.get_level_values(1), [0] * len(df.index.get_level_values(1)), 500, classes_to_colors(df), '|')

    sns.despine(left=True)

    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    return output_figure(filename)


def scatter_plot(scatter_data, df, filename=None):
    plt.figure()

    plt.scatter(scatter_data[:, 0], scatter_data[:, 1], c=classes_to_colors(df))

    sns.despine(left=True, bottom=True)

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    add_date_color_bar(df)

    return output_figure(filename)


def add_date_color_bar(df):
    num_ticks = 5
    ticker = mpl.ticker.MaxNLocator(num_ticks + 2, prune='both')

    mappable = cm.ScalarMappable(cmap=plt.cm.get_cmap('autumn'))
    mappable.set_array(range(num_ticks + 2))

    cb = plt.colorbar(mappable, ticks=ticker, shrink=0.75)
    cb.ax.set_yticklabels([df.index.values[i][1].strftime('%b %Y')
                           for i in range(0, len(df.index.values), len(df.index.values) / (num_ticks + 2))])
    cb.outline.set_linewidth(0)


def pca_loadings_table(pca, metrics):
    pca_table = pd.DataFrame(index=range(len(metrics)), columns=['Metric', 0, 1])

    pca_table['Metric'] = metrics
    pca_table[0] = pca.components_[0]
    pca_table[1] = pca.components_[1]
    pca_table.columns = ['Metric',
                         '{{PC 1 ({:.1f}\,\%)}}'.format(pca.explained_variance_ratio_[0] * 100),
                         '{{PC 2 ({:.1f}\,\%)}}'.format(pca.explained_variance_ratio_[1] * 100)]

    return pca_table.to_latex(index=False, escape=False, float_format=lambda x: '{}{:.5f}'.format(
        '\cellcolor{gray} ' if abs(x) >= 0.4 else '\cellcolor{lightgray} ' if abs(x) >= 0.2 else '', x))


def visualize_pca(df, filename=None):
    # transform data to lower dimension
    pca = PCA(2)
    pca_data = DataFrameMapper([(df.columns.values, pca)]).fit_transform(df)

    # print pca_loadings_table(pca, df.columns.values)

    # plot
    return scatter_plot(pca_data, df, filename)


def visualize_tsne(df, filename=None):
    # transform data to lower dimension
    tsne_data = TSNE(2, 15, init='pca').fit_transform(df.values)

    # plot
    return scatter_plot(tsne_data, df, filename)


def plot_outlier_score_hist(outlier_scores, score_cutoff, filename=None):
    plt.figure()

    sns.distplot(outlier_scores * 100, bins=20, kde=False, axlabel='Outlier score (%)', hist_kws={'histtype': 'stepfilled'})
    plt.ylabel('Number of experiments')

    if score_cutoff is not None:
        plt.axvline(score_cutoff * 100, color=sns.color_palette()[0], linestyle='--')

    sns.despine()

    return output_figure(filename)


def visualize_subspace_boxplots(data, highlights=None, filename=None):
    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.boxplot(data=data, orient='v', palette='Blues_d')

        if highlights is not None:
            for i in range(len(highlights)):
                plt.plot(i, highlights[i], color='red', marker='d')

        plt.xticks(rotation=30, fontsize=10)

        return output_figure(filename)


def visualize_metric_boxplot(attributes, data, strip=None, filename=None):
    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.boxplot(data=data[attributes])

        if strip is not None:
            sns.stripplot(data=strip[attributes], jitter=True, color='#C55152')

        return output_figure(filename)


def visualize_boxplot(data, strip=None, title=None, filename=None, **kwargs):
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.boxplot(data=data, **kwargs)

        if strip is not None:
            sns.stripplot(data=strip, jitter=True, color='#C55152')

        if title is not None:
            plt.title(title)

        return output_figure(filename)


def visualize_feature_importances(feature_importances, filename=None):
    feature_importances.sort(ascending=False)

    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.barplot(x=feature_importances.index.values, y=feature_importances, palette='Blues_d')

        plt.xticks(rotation='vertical', fontsize=5)

        return output_figure(filename)


def visualize_dbscan_clustering(data, cluster_labels, core_points_indices):
    plt.figure()

    # differentiate the core points
    core_samples_mask = np.zeros_like(cluster_labels, dtype=bool)
    core_samples_mask[core_points_indices] = True

    # color each cluster in a separate color (noise = black)
    unique_labels = sorted(set(cluster_labels))
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(unique_labels)))
    for label, col in zip(unique_labels, colors):
        if label == -1:
            col = 'black'

        class_member_mask = cluster_labels == label

        cluster_core_data = data[class_member_mask & core_samples_mask]
        plt.plot(cluster_core_data[:, 0], cluster_core_data[:, 1], 'o', mfc=col, mec='black', ms=14)

        cluster_other_data = data[class_member_mask & ~core_samples_mask]
        plt.plot(cluster_other_data[:, 0], cluster_other_data[:, 1], 'o', mfc=col, mec='black', ms=6)

    plt.title('Number of clusters: {}'.format(len(unique_labels) - (1 if -1 in unique_labels else 0)))

    plt.show()
    plt.close()
