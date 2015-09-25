import argparse

import fim
import pandas as pd

import outlier
import preprocess
import export


##############################################################################

# DATA LOADING AND PRE-PROCESSING

def load_metrics(file_in, min_var, min_corr):
    # load data from the input file
    data_raw = preprocess.load_metrics(file_in)

    # pre-process: remove low-variance and correlated metrics & scale the values
    data, variance, corr = preprocess.preprocess(data_raw, min_variance=min_var, min_corr=min_corr)

    # add the preprocessing results to the qcML export
    exporter.low_variance(pd.Series(variance, index=data_raw.columns.values), min_var)
    exporter.correlation(corr, min_corr)

    # add general visualizations to the qcML export
    exporter.global_visualization(data)

    return data


##############################################################################

# OUTLIER DETECTION

def detect_outliers(data, k, outlier_threshold=None, num_bins=20):
    # compute outlier scores
    outlier_scores = outlier.detect_outliers_loop(data, k)

    # compute the outlier threshold (if required)
    if outlier_threshold is None:
        outlier_threshold = outlier.detect_outlier_score_threshold(outlier_scores, num_bins)

    # add the outlier score information to the qcML export
    exporter.outlier_scores(outlier_scores, outlier_threshold, num_bins)

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
        exporter.outlier(outliers.loc[name], data)

    return data, outliers


def analyze_outliers(outliers, min_sup, min_length):
    # detect frequently occurring explanatory subspaces
    frequent_subspaces = sorted(fim.fim(outliers.Subspace, supp=min_sup, zmin=min_length), key=lambda x: x[1][0], reverse=True)
    frequent_subspaces_table = pd.DataFrame(index=range(len(frequent_subspaces)),
                                            columns=['Outlier subspace QC metric(s)', 'Number of outlying experiments'])
    for i, (subspace, (support,)) in enumerate(frequent_subspaces):
        frequent_subspaces_table.set_value(i, 'Outlier subspace QC metric(s)', ', '.join(subspace))
        frequent_subspaces_table.set_value(i, 'Number of outlying experiments', support)

    exporter.frequent_outlier_subspaces(frequent_subspaces_table, min_sup, min_length)

    return frequent_subspaces


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
    global exporter
    exporter = export.Exporter(True, True)

    data = load_metrics(args.file_in, args.min_var, args.min_corr)
    data_excluding_outliers, outliers = detect_outliers(data, args.k_neighbors, args.min_outlier, args.num_bins)
    frequent_subspaces = analyze_outliers(outliers, args.min_sup, args.min_length)

    exporter.export('out.qcml')

    import validate
    psms, inlier_psms, outlier_psms = validate.compare_outlier_psms('psms.csv', outliers)
    validate.compare_outlier_subspace_psms(outliers, frequent_subspaces, psms, inlier_psms)

# args = parse_args(None)
run(parse_args('-k 15 -o 0.99 TCGA_Quameter.tsv'.split()))

##############################################################################
