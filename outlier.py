import itertools
import numpy as np
import pandas as pd
import random
import sqlite3

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

import visualize


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


def remove_outliers(df, outlier_scores, outlier_threshold):
    outlier_idx = np.where(outlier_scores > outlier_threshold)[0]

    # retrieve outliers
    outliers = df.ix[outlier_idx].reindex()
    outliers.insert(0, 'OutlierScore', outlier_scores[outlier_idx])

    # delete outliers from the data
    df = df.drop(df.index[outlier_idx])

    return df, outliers


def interpret_outlier(data_no_outliers, outlier, filename=None):
    # print '{} has outlier score {:.2f}%'.format(outlier.name[1], outlier['OutlierScore'] * 100)

    feature_importances = outlier_subspace_explanation(data_no_outliers, outlier, 15)

    fig_features = visualize.visualize_feature_importances(feature_importances, filename=filename)

    subspace = get_relevant_subspace(feature_importances)
    fig_subspace = visualize.visualize_subspace_boxplots(data_no_outliers[subspace], outlier[subspace], filename=filename)

    return fig_features, fig_subspace


def get_outlier_subspace(data_all, outlier):
    feature_importances = outlier_subspace_explanation(data_all, outlier, 15)
    subspace = get_relevant_subspace(feature_importances)

    return subspace


def outlier_subspace_explanation(data, outlier, k, alpha=0.35):
    random.seed()
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(data.values)

    # outlier nearest neighbors
    outlier_values = outlier.drop('OutlierScore').values
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
        y = [0] * (len(ref_set_idx) + len(random_inliers_idx)) + [1] * len(outliers_sample)

        forest = RandomForestClassifier(n_estimators=100)
        forest.fit(x, y)
        importances = np.add(importances, forest.feature_importances_)

    return pd.Series(importances / repeats, index=outlier.drop('OutlierScore').index)


def get_relevant_subspace(feature_importances):
    feature_importances.sort(ascending=False)

    subspace = []
    explained_importance = 0
    max_importance = feature_importances[0]
    for i in range(len(feature_importances)):
        subspace.append(feature_importances.index.values[i])

        explained_importance += feature_importances[i]
        if explained_importance > 0.5 or max_importance * 0.6 > feature_importances[i + 1]:
            break

    return subspace


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


def get_nr_psms(filename):
    conn = sqlite3.connect(filename)

    psms = {}
    c = conn.cursor()
    for result in c.execute(
            'SELECT SS.Name, COUNT(*) FROM PeptideSpectrumMatch PSM, Spectrum S, SpectrumSource SS WHERE PSM.Spectrum = S.Id AND S.Source = SS.Id GROUP BY SS.Id'):
        psms[result[0]] = result[1]

    return pd.Series(psms)

