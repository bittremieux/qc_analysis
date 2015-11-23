import io

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn_pandas import DataFrameMapper


sns.set_context('paper')
sns.set_style('white')


def _output_figure(filename):
    out = None

    if filename is None:
        plt.show()
    elif filename == '__qcml_export__':
        binary = io.BytesIO()
        plt.savefig(binary, format='svg')
        binary.seek(0)  # rewind the data
        out = binary.read()
    else:
        plt.savefig(filename)

    plt.close()

    return out


# Remember to use the Agg matplotlib backend for the heatmap annotation to work!
def plot_correlation_matrix(corr, filename=None):
    plt.figure(figsize=(11, 10))

    # generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(corr, vmin=-1, vmax=1, linewidths=.5, square=True,
                xticklabels=corr.columns.values[:-1], yticklabels=corr.columns.values[1:],
                mask=mask, cbar_kws={'shrink': .75}, annot=True, fmt='.2f', annot_kws={'size': 4})

    # rotate overly long tick labels
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    return _output_figure(filename)


def _classes_to_colors(df):
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


def plot_timestamps(df, filename=None):
    plt.figure()

    plt.scatter(df.index.get_level_values(1), [0] * len(df.index.get_level_values(1)), 500, _classes_to_colors(df), '|')

    sns.despine(left=True)

    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    return _output_figure(filename)


def _add_date_color_bar(df):
    num_ticks = 5
    ticker = mpl.ticker.MaxNLocator(num_ticks + 2, prune='both')

    mappable = cm.ScalarMappable(cmap=plt.cm.get_cmap('autumn'))
    mappable.set_array(range(num_ticks + 2))

    cb = plt.colorbar(mappable, ticks=ticker, shrink=0.75)
    cb.ax.set_yticklabels([df.index.values[i][1].strftime('%b %Y')
                           for i in range(0, len(df.index.values), len(df.index.values) // (num_ticks + 2))])
    cb.outline.set_linewidth(0)


def _scatter_plot(scatter_data, df, filename=None):
    plt.figure()

    plt.scatter(scatter_data[:, 0], scatter_data[:, 1], c=_classes_to_colors(df))

    sns.despine(left=True, bottom=True)

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    _add_date_color_bar(df)

    return _output_figure(filename)


def plot_pca(df, filename=None):
    # transform data to lower dimension
    pca = PCA(2)
    pca_data = DataFrameMapper([(df.columns.values, pca)]).fit_transform(df)

    # plot
    return _scatter_plot(pca_data, df, filename)


def plot_tsne(df, filename=None):
    # transform data to lower dimension
    tsne_data = TSNE(2, init='pca').fit_transform(df.values)

    # plot
    return _scatter_plot(tsne_data, df, filename)


def scatter_outliers(scatter_data, df, outlier_scores, score_threshold, filename=None):
    plt.figure()

    colors = _classes_to_colors(df)

    for i, d in enumerate(scatter_data):
        if outlier_scores[i] > score_threshold:
            size = 10 + (outlier_scores[i] - score_threshold) * 200
            color = colors[i]
            marker = 'o'
            alpha = None
        else:
            size = 10
            color = 'grey'
            marker = '.'
            alpha = 0.25
        plt.scatter(d[0], d[1], s=size, c=color, marker=marker, alpha=alpha)

    sns.despine(left=True, bottom=True)

    plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    plt.tick_params(axis='y', which='both', left='off', right='off', labelleft='off')

    _add_date_color_bar(df)

    return _output_figure(filename)


def plot_pca_outliers(df, outlier_scores, score_threshold, filename=None):
    # transform data to lower dimension
    pca = PCA(2)
    pca_data = DataFrameMapper([(df.columns.values, pca)]).fit_transform(df)

    # plot
    return scatter_outliers(pca_data, df, outlier_scores, score_threshold, filename)


def plot_tsne_outliers(df, outlier_scores, score_threshold, filename=None):
    # transform data to lower dimension
    tsne_data = TSNE(2, init='pca').fit_transform(df.values)

    # plot
    return scatter_outliers(tsne_data, df, outlier_scores, score_threshold, filename)


def plot_outlier_score_hist(outlier_scores, num_bins, score_cutoff, filename=None):
    plt.figure()

    sns.distplot(outlier_scores * 100, bins=num_bins, kde=False, axlabel='Outlier score (%)', hist_kws={'histtype': 'stepfilled'})
    plt.ylabel('Number of experiments')

    if score_cutoff is not None:
        plt.axvline(score_cutoff * 100, color=sns.color_palette()[0], linestyle='--')

    sns.despine()

    return _output_figure(filename)


def plot_feature_importances(feature_importances, filename=None):
    sorted_importances = feature_importances.sort_values(ascending=False)

    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.barplot(x=sorted_importances.index.values, y=sorted_importances, palette='Blues_d')

        plt.xticks(rotation='vertical', fontsize=5)

        return _output_figure(filename)


def plot_subspace_boxplots(data, highlights=None, filename=None):
    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.boxplot(data=data, orient='v', palette='Blues_d')

        if highlights is not None:
            for i in range(len(highlights)):
                plt.plot(i, highlights[i], color='red', marker='d')

        plt.xticks(rotation=30, fontsize=10)

        return _output_figure(filename)


def plot_psm_boxplots(data, filename=None, **kwargs):
    mpl.rc('text', usetex=True)
    mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    with sns.axes_style('whitegrid'):
        fig = plt.figure()
        fig.set_tight_layout(True)

        sns.boxplot(data=data, **kwargs)

        return _output_figure(filename)


def plot_aucs(aucs, k_range, filename=None):
    max_auc = max(aucs)
    max_k = [k for k, a in zip(k_range, aucs) if a == max_auc]

    plt.figure()

    # plot all auc's
    plt.plot(k_range, aucs)
    # highlight max auc
    for k in max_k:
        plt.scatter(k, max_auc, s=50, c=sns.color_palette()[0], marker='D')

    plt.xlim(xmin=0)
    plt.ylim([0.5, 1.0])

    return _output_figure(filename)


def plot_outlier_classes_score_hist(outlier_scores, outlier_quality, num_bins, filename=None):
    # generate the histogram values for the three classes
    outlier_scores *= 100   # percentage scores
    bins = np.arange(0, 101, 100 / num_bins)
    hist_good, _ = np.histogram([score for i, score in enumerate(outlier_scores) if outlier_quality.iloc[i] == 'good'], bins=bins)
    hist_ok, _ = np.histogram([score for i, score in enumerate(outlier_scores) if outlier_quality.iloc[i] == 'ok'], bins=bins)
    hist_poor, _ = np.histogram([score for i, score in enumerate(outlier_scores) if outlier_quality.iloc[i] == 'poor'], bins=bins)
    hist = pd.DataFrame({'good': hist_good, 'ok': hist_ok, 'poor': hist_poor}, bins[:-1])

    plt.figure()

    ax = hist.plot(kind='bar', position=0)

    plt.xlabel('Outlier score (%)')
    plt.ylabel('Number of experiments')

    # change the x-axis to not include each bin value
    ax.xaxis.set_ticks(range(0, 21, 4))
    ax.xaxis.set_ticklabels(range(0, 101, 20), rotation=0)

    return _output_figure(filename)


def plot_roc(true_classes, predicted_scores, filename=None):
    # compute false positive rate and true positive rate
    fpr, tpr, _ = roc_curve(true_classes, predicted_scores)

    plt.figure()

    # plot the ROC curve
    plt.plot(fpr, tpr, label='ROC curve (AUC = {0:.2f})'.format(roc_auc_score(true_classes, predicted_scores)))

    # plot the random ROC curve at 0.5
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')

    return _output_figure(filename)


def plot_precision_recall(true_classes, predicted_scores, filename=None):
    # compute false positive rate and true positive rate
    precision, recall, _ = precision_recall_curve(true_classes, predicted_scores)

    plt.figure()

    # plot the ROC curve
    plt.plot(recall, precision, label='Precision-recall curve (average precision = {0:.2f})'
             .format(average_precision_score(true_classes, predicted_scores)))

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.legend(loc='lower right')

    return _output_figure(filename)

