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
    data = preprocess.preprocess(data_raw, min_variance=min_var, min_corr=min_corr)

    # add general visualizations to the qcML export
    qcml.add_visualization(data)

    return data


##############################################################################

# OUTLIER DETECTION

def detect_outliers(data, outlier_threshold, k):
    # read computed outlier scores
    # TODO: reimplement LoOP in Python
    outlier_scores = outlier.read_elki_outlier_scores('loop-outlier_order.txt')

    # add the outlier score information to the qcML export
    qcml.add_outlier_scores(outlier_scores, outlier_threshold)

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
    for i, (subspace, (support,)) in zip(range(len(frequent_subspaces)), frequent_subspaces):
        frequent_subspaces_table.set_value(i, 'Outlier subspace QC metric(s)', ', '.join(subspace))
        frequent_subspaces_table.set_value(i, 'Number of outlying experiments', support)

    # TODO
    # print frequent_subspaces_table.to_latex(index=False)

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
    parser.add_argument('--min_outlier', '-o', type=float)
    parser.add_argument('--min_sup', '-s', default=5, type=int)
    parser.add_argument('--min_length', '-l', default=1, type=int)

    # parse command-line arguments
    return parser.parse_args(args)


def run(args):
    data = load_metrics(args.file_in, args.min_var, args.min_corr)
    data_excluding_outliers, outliers = detect_outliers(data, args.min_outlier, args.k_neighbors)
    analyze_outliers(outliers, args.min_sup, args.min_length)

    qcml.export('out.qcml')


qcml = qcml_export.QcmlExport()

# args = parse_args(None)
run(parse_args('-k 15 -o 0.98 TCGA_Quameter.tsv'.split()))

##############################################################################
