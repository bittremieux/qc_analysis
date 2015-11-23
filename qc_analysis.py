import argparse
import math
import multiprocessing
import shlex

import numpy as np
import pandas as pd
import pymining.itemmining as im
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

import export
import outlier
import preprocess


##############################################################################

# DATA LOADING AND PRE-PROCESSING

def load_metrics(file_in, min_var, min_corr, scaling):
    # load data from the input file
    data_raw = preprocess.load_metrics(file_in)

    # pre-process: remove low-variance and correlated metrics & scale the values
    data, variance, corr = preprocess.preprocess(data_raw, min_variance=min_var, min_corr=min_corr, scaling_mode=scaling)

    # add the preprocessing results to the qcML export
    exporter.low_variance(pd.Series(variance, index=data_raw.columns.values), min_var)
    exporter.correlation(corr, min_corr)
    exporter.preprocess_overview(data_raw.columns.values, variance, min_var, corr, min_corr)

    # add general visualizations to the qcML export
    exporter.global_visualization(data)

    return data


##############################################################################

# OUTLIER DETECTION

def detect_outliers(data, k, dist, outlier_threshold=None, num_bins=20):
    # compute outlier scores
    outlier_scores = outlier.detect_outliers_loop(data, k, metric=dist)

    # compute the outlier threshold (if required)
    if outlier_threshold is None:
        outlier_threshold = outlier.detect_outlier_score_threshold(outlier_scores, num_bins)

    # add the outlier score information to the qcML export
    exporter.outlier_scores(data, outlier_scores, outlier_threshold, num_bins)

    # separate significant outliers
    data_excluding_outliers, outliers = outlier.split_outliers(data, outlier_scores, outlier_threshold)

    return outliers, outlier_scores


def analyze_outliers(data, outliers, k, min_sup):
    # retrieve explanatory subspaces for each outlier
    outliers['FeatureImportance'] = object
    outliers['Subspace'] = object
    with multiprocessing.Pool() as pool:
        # compute the subspace for each outlier
        subspaces = {name: pool.apply_async(outlier.get_outlier_subspace, (data, this_outlier, k))
                     for name, this_outlier in outliers.iterrows()}

        # set the outlier subspaces
        for name, result in subspaces.items():
            feature_importance, subspace = result.get()
            outliers.set_value(name, 'FeatureImportance', feature_importance.values)
            outliers.set_value(name, 'Subspace', subspace)

    # add the outliers' subspaces to the export
    for name, this_outlier in outliers.iterrows():
        exporter.outlier(this_outlier, data)

    # detect frequently occurring explanatory subspaces
    abs_sup = min_sup * -1 if min_sup < 0 else min_sup * len(outliers) // 100
    frequent_subspaces = sorted(im.relim(im.get_relim_input(outliers.Subspace), min_support=abs_sup).items(), key=lambda x: x[1], reverse=True)
    frequent_subspaces_table = pd.DataFrame(index=range(len(frequent_subspaces)),
                                            columns=['Outlier subspace QC metric(s)', 'Number of outlying experiments'])
    for i, (subspace, support) in enumerate(frequent_subspaces):
        frequent_subspaces_table.set_value(i, 'Outlier subspace QC metric(s)', ', '.join(subspace))
        frequent_subspaces_table.set_value(i, 'Number of outlying experiments', support)

    exporter.frequent_outlier_subspaces(frequent_subspaces_table, min_sup)

    return frequent_subspaces


##############################################################################

# OUTLIER VALIDATION BY PSM COMPARISON

def compare_outlier_psms(f_psms, outliers):
    # compare inliers and outliers based on their number of valid PSM's
    psms = pd.Series.from_csv(f_psms)

    outlier_psms = psms.filter(items=[index[0] for index in outliers.index.values])
    inlier_psms = psms.drop(outlier_psms.index)

    exporter.psm(inlier_psms, outlier_psms)

    return psms, inlier_psms, outlier_psms


def compare_outlier_subspace_psms(outliers, frequent_subspaces, psms, inlier_psms):
    # test whether a subspace can be related to a lower number of PSM's
    psm_table = pd.DataFrame(index=psms.index)
    psm_table['\\bfseries Inliers'] = inlier_psms
    pval_table = pd.DataFrame(index=range(len(frequent_subspaces)), columns=['Metric(s)', 'Number of outliers', '\emph{p}-value'])
    for i, (subspace, support) in enumerate(frequent_subspaces):
        subspace = sorted(subspace)

        # compare outlier values
        outliers_values = pd.DataFrame([this_outlier for _, this_outlier in outliers.iterrows() if set(subspace) <= set(this_outlier.Subspace)])

        # compare outlier PSM's
        outlier_psms = psms.filter(items=[index[0] for index in outliers_values.index.values])

        # quantify difference between inliers and outliers
        t_stat, p_value = stats.ttest_ind(inlier_psms.values, outlier_psms.values, equal_var=False)

        psm_table['{}{}'.format('\\itshape ' if p_value <= 0.05 else '', ', '.join(subspace))] = outlier_psms

        pval_table.set_value(i, 'Metric(s)', ', '.join(subspace))
        pval_table.set_value(i, 'Number of outliers', support)
        pval_table.set_value(i, '\emph{p}-value', p_value)

    exporter.psm_pval(psm_table, pval_table)


# OUTLIER VALIDATION USING PNNL EXPERT CLASSIFICATION

def _quality_classes_to_binary(filenames, quality_classes):
    # convert quality classes: 1 -> poor, 0 -> good/ok
    # requires that NO unvalidated samples are present
    return [1 if quality_classes[f] == 'poor' else 0 for f in filenames]


def find_optimal_outliers_k(data, f_class, k_min, dist):
    true_classes = _quality_classes_to_binary(data.index.get_level_values(0), pd.Series.from_csv(f_class))
    k_range = np.arange(k_min, math.ceil(len(data) / 2), dtype=int)

    aucs = []
    for k in k_range:
        outlier_scores = outlier.detect_outliers_loop(data, k, metric=dist)
        aucs.append(roc_auc_score(true_classes, outlier_scores))
    max_auc = max(aucs)
    max_k = [k for k, auc in zip(k_range, aucs) if auc == max_auc]

    exporter.outlier_auc(aucs, k_range)

    return max_k, max_auc


def validate_outlier_score(data, f_class, scores, num_bins=20):
    quality_classes = pd.Series.from_csv(f_class)
    true_classes = _quality_classes_to_binary(data.index.get_level_values(0), quality_classes)

    exporter.outlier_validation(scores, quality_classes, num_bins, true_classes)


##############################################################################

#  EXECUTE

# command-line execution
def parse_args(args_str=None):
    parser = argparse.ArgumentParser(description='Mass spectrometry quality control metrics analysis')
    parser.add_argument('file_in', type=argparse.FileType('r'),
                        help='the tab-separated input file containing the QC metrics')
    parser.add_argument('file_out', type=argparse.FileType('w'),
                        help='the name of the qcML output file')
    parser.add_argument('--min_var', '-var', default=0.0001, type=float,
                        help='metrics with a lower variance will be removed (default: %(default)s)')
    parser.add_argument('--min_corr', '-corr', default=0.9, type=float,
                        help='metrics with a higher correlation will be removed (default: %(default)s)')
    parser.add_argument('--scaling_mode', '-scale', default='robust', type=str, choices=['robust', 'standard'],
                        help='mode to standardize the metric values (default: %(default)s)')
    parser.add_argument('--k_neighbors', '-k', type=int, required=True,
                        help='the number of nearest neighbors used for outlier detection')
    parser.add_argument('--distance', '-dist', default='manhattan', type=str,
                        help='metric to use for distance computation (default: %(default)s) '
                             'ny metric from scikit-learn or scipy.spatial.distance can be used')
    parser.add_argument('--min_outlier', '-o', default=None, type=float,
                        help='the minimum outlier score threshold (default: %(default)s) '
                             'if no threshold is provided, an automatic threshold is determined')
    parser.add_argument('--num_bins', '-bin', default=20, type=int,
                        help='the number of bins for the outlier score histogram (default: %(default)s)')
    parser.add_argument('--min_sup', '-sup', default=5, type=int,
                        help='the minimum support for subspace frequent itemset mining (default: %(default)s) '
                             'positive numbers are interpreted as percentages, negative numbers as absolute supports')

    # parse command-line arguments
    if args_str is None:
        return parser.parse_args()
    # or parse string arguments
    else:
        return parser.parse_args(shlex.split(args_str))


def run(args):
    global exporter
    exporter = export.Exporter(True, False)

    data = load_metrics(args.file_in, args.min_var, args.min_corr, args.scaling_mode)
    outliers, outliers_score = detect_outliers(data, args.k_neighbors, args.distance, args.min_outlier, args.num_bins)
    analyze_outliers(data, outliers, args.k_neighbors, args.min_sup)

    exporter.export(args.file_out)


def generate_images(args, f_psms=None, f_class=None, k_min=2):
    global exporter
    exporter = export.Exporter(False, True)

    data = load_metrics(args.file_in, args.min_var, args.min_corr, args.scaling_mode)

    # compare outliers based on the number of psm's
    if f_psms is not None:
        outliers, outliers_score = detect_outliers(data, args.k_neighbors, args.distance, args.min_outlier, args.num_bins)
        frequent_subspaces = analyze_outliers(data, outliers, args.k_neighbors, args.min_sup)
        psms, inlier_psms, outlier_psms = compare_outlier_psms(f_psms, outliers)
        compare_outlier_subspace_psms(outliers, frequent_subspaces, psms, inlier_psms)

    # compare outliers based on manual expert evaluation
    if f_class is not None:
        optimal_ks, _ = find_optimal_outliers_k(data, f_class, k_min, args.distance)
        outliers, outliers_score = detect_outliers(data, optimal_ks[0], args.distance, args.min_outlier, args.num_bins)
        validate_outlier_score(data, f_class, outliers_score, args.num_bins)


if __name__ == '__main__':
    run(parse_args())


##############################################################################
