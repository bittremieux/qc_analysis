import argparse

import fim
import numpy as np
import pandas as pd

import outlier
import preprocess
import qcml_export


##############################################################################

# DATA LOADING AND PRE-PROCESSING

def load_metrics(file_in, min_var, min_corr):
    # load data from the input file
    data_raw = preprocess.load_metrics(file_in)

    # pre-process: remove low-variance and correlated metrics & scale the values
    data, variance, corr = preprocess.preprocess(data_raw, min_variance=min_var, min_corr=min_corr)

    # add the preprocessing results to the qcML export
    qcml.add_low_variance(pd.Series(variance, index=data_raw.columns.values), min_var)
    qcml.add_correlation(corr, min_corr)

    # add general visualizations to the qcML export
    qcml.add_visualization(data)

    # optional: correlation matrix & PCA loadings
    # visualize.visualize_correlation_matrix(corr)
    # print pca_loadings_table(pca, df.columns.values)

    return data


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


##############################################################################

# OUTLIER DETECTION

def detect_outliers(data, k, outlier_threshold=None, num_bins=20):
    # compute outlier scores
    outlier_scores = outlier.detect_outliers_loop(data, k)

    # compute the outlier threshold (if required)
    if outlier_threshold is None:
        outlier_threshold = outlier.detect_outlier_score_threshold(outlier_scores, num_bins)

    # add the outlier score information to the qcML export
    qcml.add_outlier_scores(outlier_scores, outlier_threshold, num_bins)

    # remove significant outliers
    data_including_outliers = data
    data, outliers = outlier.split_outliers(data, outlier_scores, outlier_threshold)

    # retrieve explanatory subspaces for each outlier
    outliers['FeatureImportance'] = object
    outliers['Subspace'] = object
    for name, this_outlier in outliers.iterrows():
        feature_importance, subspace = outlier.get_outlier_subspace(data_including_outliers, this_outlier, k)
        outliers.set_value(name, 'FeatureImportance', feature_importance.values)
        outliers.set_value(name, 'Subspace', subspace)

        # add the outlier to the qcML export
        qcml.add_outlier_runquality(outliers.loc[name], data)

    return data, outliers


def analyze_outliers(outliers, min_sup, min_length):
    # detect frequently occurring explanatory subspaces
    frequent_subspaces = sorted(fim.fim(outliers.Subspace, supp=min_sup, zmin=min_length), key=lambda x: x[1][0], reverse=True)
    frequent_subspaces_table = pd.DataFrame(index=range(len(frequent_subspaces)),
                                            columns=['Outlier subspace QC metric(s)', 'Number of outlying experiments'])
    for i, (subspace, (support,)) in enumerate(frequent_subspaces):
        frequent_subspaces_table.set_value(i, 'Outlier subspace QC metric(s)', ', '.join(subspace))
        frequent_subspaces_table.set_value(i, 'Number of outlying experiments', support)

    qcml.add_frequent_outlier_subspaces(frequent_subspaces_table, min_sup, min_length)

    # print frequent_subspaces_table.to_latex(index=False)

    # TODO
    # # compare inliers and outliers based on their number of valid PSM's
    # psms = outlier.get_nr_psms('/Volumes/BADReM Wout/TCGA_Zhang/idp/VU_N95_PP.idpDB')
    #
    # outlier_psms = psms.filter(items=[index[1] for index in outliers.index.values])
    # inlier_psms = psms.drop(outlier_psms.index)
    # visualize.visualize_boxplot(pd.DataFrame({'Inliers ({})'.format(len(inlier_psms)): inlier_psms,
    #                                           'Outliers ({})'.format(len(outlier_psms)): outlier_psms}),
    #                             filename='psm_all.pdf')
    #
    # # test whether a subspace can be related to a lower number of PSM's
    # psm_table = pd.DataFrame(index=psms.index)
    # psm_table['\\bfseries Inliers'] = inlier_psms
    # pval_table = pd.DataFrame(index=range(len(frequent_subspaces)),
    #                           columns=['Metric(s)', 'Number of outliers', '\emph{p}-value'])
    # for i, (subspace, (support,)) in zip(range(len(frequent_subspaces)), frequent_subspaces):
    #     subspace = map(None, subspace)
    #
    #     # compare outlier values
    #     outliers_values = pd.DataFrame([this_outlier for _, this_outlier in outliers.iterrows()
    #                                     if set(subspace) <= set(this_outlier.Subspace)])
    #
    #     # visualize.visualize_metric_boxplot(subspace, data, outliers_values)
    #
    #     # compare outlier PSM's
    #     outlier_psms = psms.filter(items=[index[1] for index in outliers_values.index.values])
    #
    #     # quantify difference between inliers and outliers
    #     import scipy.stats as stats     # TODO: move
    #     t_stat, p_value = stats.ttest_ind(inlier_psms.values, outlier_psms.values, equal_var=False)
    #
    #     psm_table['{}{}'.format('\\itshape ' if p_value <= 0.05 else '', ', '.join(subspace))] = outlier_psms
    #
    #     pval_table.set_value(i, 'Metric(s)', ', '.join(subspace))
    #     pval_table.set_value(i, 'Number of outliers', support)
    #     pval_table.set_value(i, '\emph{p}-value', p_value)
    #
    #     # visualize.visualize_boxplot(pd.DataFrame({'Inliers ({})'.format(len(inlier_psms)): inlier_psms,
    #     #                                           'Outliers ({})'.format(len(outlier_psms)): outlier_psms}),
    #     #                             title=', '.join(subspace), filename='{}.pdf'.format('_'.join(subspace)), orient='v')
    #
    # print pval_table.to_latex(index=False, escape=False,
    #                           float_format=lambda x: '{}{:.5f}'.format('\cellcolor{lightgray} ' if x <= 0.05 else '', x))
    #
    # visualize.visualize_boxplot(psm_table, filename='psm_subspace.pdf', orient='h')


# def visualize_psm_boxplot(data, filename=None, **kwargs):
#     mpl.rc('text', usetex=True)
#     mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
#
#     with sns.axes_style('whitegrid'):
#         fig = plt.figure()
#         fig.set_tight_layout(True)
#
#         sns.boxplot(data=data, **kwargs)
#
#         return output_figure(filename)

##############################################################################

#  EXECUTE

# if __name__ == '__main__':


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Quality control metrics outlier detection', epilog='Citation information')
    parser.add_argument('file_in', type=argparse.FileType('r'))
    # TODO
    # parser.add_argument('file_out', type=argparse.FileType('w'), required=False)
    parser.add_argument('--min_var', '-v', default=0.0001, type=float)
    parser.add_argument('--min_corr', '-c', default=0.9, type=float)
    parser.add_argument('--k_neighbors', '-k', type=int)
    parser.add_argument('--min_outlier', '-o', default=None, type=float)
    parser.add_argument('--num_bins', '-b', default=20, type=int)
    parser.add_argument('--min_sup', '-s', default=5, type=int)
    parser.add_argument('--min_length', '-l', default=1, type=int)

    # parse command-line arguments
    return parser.parse_args(args)


def run(args):
    global qcml
    qcml = qcml_export.QcmlExport()

    data = load_metrics(args.file_in, args.min_var, args.min_corr)
    data_excluding_outliers, outliers = detect_outliers(data, args.k_neighbors, args.min_outlier, args.num_bins)
    analyze_outliers(outliers, args.min_sup, args.min_length)

    qcml.export('out.qcml')


qcml = None

# args = parse_args(None)
run(parse_args('-k 15 -s 10 TCGA_Quameter.tsv'.split()))

##############################################################################
