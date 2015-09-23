import itertools
import numpy as np
import pandas as pd
import random
import sqlite3

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

import visualize


random.seed()


# TODO
def read_elki_outlier_scores(filename):
    f = open(filename, 'r')

    scores = {}
    for line in f:
        line = line.strip()
        idx = int(line[line.index('ID=') + len('ID=') : line.index(' ')].strip())
        score = float(line[line.index('outlier=') + len('outlier='):].strip())
        scores[idx] = score

    f.close()

    return np.array([scores[i] for i in sorted(scores)])


def split_outliers(data, outlier_scores, outlier_threshold):
    # identify the outliers with a score exceeding the threshold
    outlier_idx = np.where(outlier_scores > outlier_threshold)[0]
    # select the outliers and add their scores
    outliers = data.iloc[outlier_idx].reindex()
    outliers.insert(len(outliers.columns), 'OutlierScore', outlier_scores[outlier_idx])
    # sort the outliers based on their scores
    outliers.sort('OutlierScore', ascending=False, inplace=True)

    # remove the outliers from the data
    data_excluding_outliers = data.drop(data.index[outlier_idx])

    return data_excluding_outliers, outliers


def get_outlier_subspace(data, outlier, k):
    feature_importances = outlier_subspace_explanation(data, outlier, k)
    subspace = get_relevant_subspace(feature_importances)

    return feature_importances, subspace


def outlier_subspace_explanation(data, outlier, k, alpha=0.35):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(data.values)

    # outlier nearest neighbors
    outlier_values = outlier.drop(['OutlierScore', 'FeatureImportance', 'Subspace']).values
    k_dist = knn.kneighbors(outlier_values)[0][0][-1]
    ref_set_idx = knn.radius_neighbors(outlier_values, k_dist)[1][0]

    # distribution to supersample the outlier
    l = alpha * k_dist / np.sqrt(len(outlier_values))
    cov = l * l * np.identity(len(outlier_values))

    repeats = 10
    importances = np.zeros(len(outlier_values), float)
    for i in range(repeats):
        # subsample random inliers
        random_inliers_idx = random.sample([x for x in range(len(data)) if x not in ref_set_idx], len(ref_set_idx))

        # supersample the outlier
        outliers_sample = np.random.multivariate_normal(outlier_values, cov, len(ref_set_idx) + len(random_inliers_idx))

        # determine the relevant features by classifying the inliers versus the outliers
        x = np.vstack([[data.iloc[i] for i in itertools.chain(ref_set_idx, random_inliers_idx)], outliers_sample])
        y = np.concatenate((np.zeros(len(ref_set_idx) + len(random_inliers_idx)), np.ones(len(outliers_sample))))

        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(x, y)
        importances = np.add(importances, forest.feature_importances_)

    return pd.Series(importances / repeats, index=outlier.drop(['OutlierScore', 'FeatureImportance', 'Subspace']).index)


def get_relevant_subspace(feature_importances):
    feature_importances.sort(ascending=False)

    subspace = []
    explained_importance = 0
    min_importance = feature_importances[0] * 2 / 3
    for i in range(len(feature_importances)):
        subspace.append(feature_importances.index.values[i])

        explained_importance += feature_importances[i]
        if explained_importance > 0.5 or i < len(feature_importances) - 1 and feature_importances[i + 1] < min_importance:
            break

    return np.array(subspace, dtype=object)


# TODO
def get_inlier_outlier_nr_psms(filename, inlier_names):
    conn = sqlite3.connect(filename)

    inlier_psms = {}
    outlier_psms = {}
    c = conn.cursor()
    for result in c.execute('SELECT SS.Name, COUNT(*) FROM PeptideSpectrumMatch PSM, Spectrum S, SpectrumSource SS WHERE PSM.Spectrum = S.Id AND S.Source = SS.Id GROUP BY SS.Id'):
        if result[0] in inlier_names:
            inlier_psms[result[0]] = result[1]
        else:
            outlier_psms[result[0]] = result[1]

    return pd.Series(inlier_psms), pd.Series(outlier_psms)


# TODO
def get_nr_psms(filename):
    conn = sqlite3.connect(filename)

    psms = {}
    c = conn.cursor()
    for result in c.execute(
            'SELECT SS.Name, COUNT(*) FROM PeptideSpectrumMatch PSM, Spectrum S, SpectrumSource SS WHERE PSM.Spectrum = S.Id AND S.Source = SS.Id GROUP BY SS.Id'):
        psms[result[0]] = result[1]

    return pd.Series(psms)
