import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler

import qcml_export


def load_metrics(file_in):
    metrics = pd.read_csv(file_in, '\t', index_col=0, parse_dates=[1])
    metrics.set_index('StartTimeStamp', append=True, inplace=True)

    return metrics


def preprocess(data, min_variance, min_corr):
    # remove low-variance and correlated metrics
    data = remove_low_variance_features(data, min_variance)
    data = remove_correlated_features(data, min_corr)
    # scale the values
    data = scale(data)

    # sort the experiments chronologically
    data.sortlevel(level=1, inplace=True)

    return data


def remove_low_variance_features(data, min_variance):
    variance_threshold = VarianceThreshold(min_variance).fit(data)

    # TODO
    # if verbose:
    #     for i in np.where(variance_threshold.variances_ <= min_variance)[0]:
    #         print '{} has low variance: {}'.format(df.columns.values[i], variance_threshold.variances_[i])

    return data[np.where(variance_threshold.variances_ > min_variance)[0]]


def remove_correlated_features(data, min_corr):
    corr = data.corr()

    # TODO
    # fig_corr = visualize.visualize_correlation_matrix(corr, filename='raw' if not verbose else None)

    # find strongly correlated features
    remove_features = set()
    for row in range(len(corr.index)):
        if corr.columns.values[row] not in remove_features:
            for col in range(row + 1, len(corr.columns)):
                if corr.columns.values[col] not in remove_features and abs(corr.ix[row, col]) > min_corr:
                    remove_features.add(corr.columns.values[col])

                    # TODO
                    # if verbose:
                    #     print '{} correlated with {}: {}'.format(
                    #         corr.columns.values[row], corr.columns.values[col], corr.ix[row, col])

    # remove the correlated features
    for feature in remove_features:
        del data[feature]

    return data


def scale(data):
    return pd.DataFrame(RobustScaler().fit_transform(data), index=data.index, columns=data.columns)
