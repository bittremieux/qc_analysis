import numpy as np
import pandas as pd

import visualize

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler


def load_metrics(quameter_file=None, imondb_file=None):
    quameter_metrics = pd.read_csv(quameter_file, '\t', index_col=0, parse_dates=[1]) if quameter_file is not None else None
    imondb_metrics = pd.read_csv(imondb_file, '\t', index_col=0, parse_dates=[1]) if imondb_file is not None else None

    if quameter_metrics is not None and imondb_metrics is not None:
        df = pd.concat([quameter_metrics, imondb_metrics.drop(['StartTimeStamp'], axis=1)], axis=1)
    else:
        df = quameter_metrics if quameter_metrics is not None else imondb_metrics

    # convert the file names to sample classes
    samples = []
    for filename in df.index.values:
        samples.append(filename[5: 19])
    df.insert(0, 'Sample', samples)

    df.set_index(['Sample', 'StartTimeStamp'], append=True, inplace=True)
    df.index = df.index.swaplevel('Filename', 'Sample')

    return df


def preprocess(df, **kwargs):
    df = remove_low_variance_features(df, **kwargs)
    df, fig_corr = remove_correlated_features(df, **kwargs)
    df = scale(df)

    return df, fig_corr


def remove_low_variance_features(df, min_variance=0.0, verbose=False, **kwargs):
    variance_threshold = VarianceThreshold(min_variance).fit(df)

    if verbose:
        for i in np.where(variance_threshold.variances_ <= min_variance)[0]:
            print '{} has low variance: {}'.format(df.columns.values[i], variance_threshold.variances_[i])

    return df[np.where(variance_threshold.variances_ > min_variance)[0]]


def remove_correlated_features(df, min_corr=0.9, verbose=False, **kwargs):
    corr = df.corr()

    fig_corr = visualize.visualize_correlation_matrix(corr, filename='raw' if not verbose else None)

    # find strongly correlated features
    remove_features = set()
    for row in range(len(corr.index)):
        if corr.columns.values[row] not in remove_features:
            for col in range(row + 1, len(corr.columns)):
                if corr.columns.values[col] not in remove_features and abs(corr.ix[row, col]) > min_corr:
                    remove_features.add(corr.columns.values[col])

                    if verbose:
                        print '{} correlated with {}: {}'.format(
                            corr.columns.values[row], corr.columns.values[col], corr.ix[row, col])

    # delete the correlated features
    for feature in remove_features:
        del df[feature]

    return df, fig_corr


def scale(df):
    return pd.DataFrame(RobustScaler().fit_transform(df), index=df.index, columns=df.columns)
