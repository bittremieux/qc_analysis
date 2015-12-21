import math

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

import export
import outlier
import qc_analysis


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
    color_classes = [0]
    pval_table = pd.DataFrame(index=range(len(frequent_subspaces)),
                              columns=['Metric(s)', 'Support (\%)', '\emph{p}-value'])
    for i, (subspace, support) in enumerate(frequent_subspaces):
        subspace = sorted(subspace)

        # compare outlier values
        outliers_values = pd.DataFrame([this_outlier for _, this_outlier in outliers.iterrows()
                                        if set(subspace) <= set(this_outlier.Subspace)])

        # compare outlier PSM's
        outlier_psms = psms.filter(items=[index[0] for index in outliers_values.index.values])

        # quantify difference between inliers and outliers
        t_stat, p_value = stats.ttest_ind(inlier_psms.values, outlier_psms.values, equal_var=False)

        psm_table['{}{}'.format('\\itshape ' if p_value <= 0.05 and t_stat > 0 else '', ', '.join(subspace))] = outlier_psms
        color_classes.append(2 if p_value <= 0.05 and t_stat > 0 else 1)

        pval_table.set_value(i, 'Metric(s)', ', '.join(subspace))
        pval_table.set_value(i, 'Support (\%)', round(support))
        pval_table.set_value(i, '\emph{p}-value', '{}{:.5f}'.format('\cellcolor{lightgray} '
                                                                    if p_value <= 0.05 and t_stat > 0 else '', p_value))

    exporter.psm_pval(psm_table, pval_table, color_classes)


# OUTLIER VALIDATION USING PNNL EXPERT CLASSIFICATION

def _quality_classes_to_binary(filenames, quality_classes):
    # convert quality classes: 1 -> poor, 0 -> good/ok
    # requires that NO unvalidated samples are present
    return np.array([1 if quality_classes[f] == 'poor' else 0 for f in filenames])


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


def run(args, f_psms=None, f_class=None, k_min=2, folder=None):
    global exporter
    exporter = export.Exporter(False, True, folder)
    qc_analysis.exporter = exporter

    data = qc_analysis.load_metrics(args.file_in, args.min_var, args.min_corr, args.scaling_mode)

    # compare outliers based on the number of psm's
    if f_psms is not None:
        outliers, outliers_score = qc_analysis.detect_outliers(data, args.k_neighbors, args.distance,
                                                               args.min_outlier, args.num_bins)
        frequent_subspaces = qc_analysis.analyze_outliers(data, outliers, args.k_neighbors, args.min_sup)
        psms, inlier_psms, outlier_psms = compare_outlier_psms(f_psms, outliers)
        compare_outlier_subspace_psms(outliers, frequent_subspaces, psms, inlier_psms)

    # compare outliers based on manual expert evaluation
    if f_class is not None:
        optimal_ks, _ = find_optimal_outliers_k(data, f_class, k_min, args.distance)
        outliers, outliers_score = qc_analysis.detect_outliers(data, optimal_ks[0], args.distance,
                                                               args.min_outlier, args.num_bins)
        validate_outlier_score(data, f_class, outliers_score, args.num_bins)
        qc_analysis.analyze_outliers(data, outliers, optimal_ks[0], args.min_sup)

    exporter.export(args.file_out)


if __name__ == '__main__':
    # PNNL
    instruments = [('iontrap', 0.15), ('orbi', 0.25), ('velos', 0.20)]
    for instrument, outlier_score in instruments:
        run(qc_analysis.parse_args('-k 1 -o {} data/PNNL_{}_QuaMeter.tsv out.qcml'.format(outlier_score, instrument)),
            f_class='data/PNNL_{}_validation.csv'.format(instrument), folder='out/{}'.format(instrument))

    # TCGA
    run(qc_analysis.parse_args('-k 50 -o 0.25 data/TCGA_QuaMeter.tsv out.qcml'), f_psms='data/TCGA_psms.csv', folder='out/tcga')


##############################################################################