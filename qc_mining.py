import pandas as pd
import fim

import preprocess
import visualize
import outlier

##############################################################################

# DATA LOADING AND PRE-PROCESSING

# load data
data_raw = preprocess.load_metrics('TCGA_QuaMeter.tsv')
# data_raw = preprocess.load_metrics('TCGA_QuaMeter.tsv', 'TCGA_iMonDB_median_statuslog.tsv')

# pre-process
data, fig_corr = preprocess.preprocess(data_raw, min_variance=0.0001, min_corr=0.9)

# sort experiments chronologically
data.sortlevel(level=2, inplace=True)

# visualize.visualize_timestamps(data, filename='tcga_dates.pdf')
# visualize.visualize_pca(data, filename='pca.pdf')
# visualize.visualize_tsne(data, filename='tsne.pdf')
fig_time = visualize.visualize_timestamps(data, filename='raw')
fig_pca = visualize.visualize_pca(data, filename='raw')
fig_tsne = visualize.visualize_tsne(data, filename='raw')

# export preprocessed metrics for external analysis (only retain raw values)
# np.savetxt('export.tsv', data, fmt='%f', delimiter='\t')
# data.to_csv('out.txt', float_format='%f', header=True, index=False)

##############################################################################

# OUTLIER DETECTION

# read computed outlier scores
outlier_scores = outlier.read_elki_outlier_scores('loop-outlier_order.txt')
outlier_threshold = 0.95
# visualize.plot_outlier_score_hist(outlier_scores, outlier_threshold, filename='outlier-hist.pdf')

# remove significant outliers
data_including_outliers = data
data, outliers = outlier.remove_outliers(data, outlier_scores, outlier_threshold)

# print '{} outliers removed with outlier score above {}'.format(len(outliers), outlier_threshold)
# visualize.visualize_timestamps(outliers)

# sort outliers based on their score
sorted_outliers = outliers.sort('OutlierScore', ascending=False)

# this_outlier = sorted_outliers.xs('TCGA-AG-A02X-01A-32_W_VU_20130213_A0218_10D_R_FR06',
#                                   level='Filename', drop_level=False).iloc[0]
# outlier.interpret_outlier(data, this_outlier, 'TCGA-AG-A02X-01A-32_W_VU_20130213_A0218_10D_R_FR06')

# # interpret outliers individually
# # for _, this_outlier in sorted_outliers.iterrows():
# #     outlier.interpret_outlier(data_including_outliers, data, this_outlier)
#
# # retrieve explanatory subspaces for each outlier
# subspaces = []
# for _, this_outlier in sorted_outliers.iterrows():
#     subspace = outlier.get_outlier_subspace(data_including_outliers, this_outlier)
#     subspaces.append(subspace)
# sorted_outliers['Subspace'] = subspaces
#
# # detect frequently occurring subspaces
# min_sup = 10
# min_length = 1
# frequent_subspaces = sorted(fim.fim(sorted_outliers.Subspace, supp=-1 * min_sup, zmin=min_length),
#                             key=lambda x: x[1][0], reverse=True)
# frequent_subspaces_table = pd.DataFrame(index=range(len(frequent_subspaces)),
#                                         columns=['Outlier subspace QC metric(s)', 'Number of outlying experiments'])
# for i, (subspace, (support,)) in zip(range(len(frequent_subspaces)), frequent_subspaces):
#     frequent_subspaces_table.set_value(i, 'Outlier subspace QC metric(s)', ', '.join(subspace))
#     frequent_subspaces_table.set_value(i, 'Number of outlying experiments', support)
#
# print frequent_subspaces_table.to_latex(index=False)
#
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
#     outliers_values = pd.DataFrame([this_outlier for _, this_outlier in sorted_outliers.iterrows()
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

# EXPORT TO QCML

import qcml
import datetime

# create qcML object
qcml_out = qcml.qcMLType()
qcml_out.set_version('0.0.8')

# add references to the controlled vocabularies (CV's)
cv_ms = qcml.CVType('PSI-MS', '3.78.0', 'http://psidev.cvs.sourceforge.net/viewvc/psidev/psi/psi-ms/mzML/controlledVocabulary/psi-ms.obo', 'MS')
cv_qc = qcml.CVType('MS-QC', '0.1.1', 'https://github.com/qcML/qcML-development/blob/master/cv/qc-cv.obo', 'QC')
cv_quameter = qcml.CVType('QuaMeter', '0', 'http://pubs.acs.org/doi/abs/10.1021/ac300629p', 'QM')
qcml_out.set_cvList(qcml.CVListType([cv_ms, cv_qc, cv_quameter]))

# add global information as a setQuality
set_quality = qcml.SetQualityAssessmentType(ID='SetQuality')
qcml_out.add_setQuality(set_quality)

# PCA & t-SNE visualization
ap_time = qcml.AttachmentType(name='Experiment execution time', ID='time', binary=fig_time)
set_quality.add_attachment(ap_time)
ap_pca = qcml.AttachmentType(name='PCA visualization', ID='PCA', binary=fig_pca)
set_quality.add_attachment(ap_pca)
ap_tsne = qcml.AttachmentType(name='t-SNE visualization', ID='t-SNE', binary=fig_tsne)
set_quality.add_attachment(ap_tsne)

# QC metrics
ap_corr = qcml.AttachmentType(name='Correlation matrix', ID='corr', binary=fig_corr)
set_quality.add_attachment(ap_corr)

# outlier histogram
qp_threshold = qcml.QualityParameterType(name='Outlier score threshold', value=outlier_threshold, ID='OutlierScoreThreshold')
set_quality.add_qualityParameter(qp_threshold)
ap_outlier_histogram = qcml.AttachmentType(name='Outlier score histogram', ID='OutlierScoreHistogram',
                                           qualityParameterRef=qp_threshold.get_ID(),
                                           binary=visualize.plot_outlier_score_hist(
                                               outlier_scores, outlier_threshold, filename='raw'))
set_quality.add_attachment(ap_outlier_histogram)

# add each outlier as a RunQuality
for _, this_outlier in sorted_outliers.iterrows():
    run_quality = qcml.RunQualityAssessmentType(ID=this_outlier.name[1])
    qcml_out.add_runQuality(run_quality)

    creation_date = qcml.MetaDataType(name='Creation date', value=datetime.datetime.now(), cvRef=cv_qc.get_ID(),
                                      accession='MS:1000747', ID='{}_CreationDate'.format(run_quality.get_ID()))
    run_quality.add_metaDataParameter(creation_date)

    qp_score = qcml.QualityParameterType(name='Outlier score', value=this_outlier['OutlierScore'],
                                         ID='{}_OutlierScore'.format(run_quality.get_ID()))
    run_quality.add_qualityParameter(qp_score)

    fig_features, fig_subspace = outlier.interpret_outlier(data, this_outlier, filename='raw')
    ap_feature_importance = qcml.AttachmentType(name='Feature importance', ID='{}_FeatureImportance'.format(
        run_quality.get_ID()), qualityParameterRef=qp_score.get_ID(), binary=fig_features)
    run_quality.add_attachment(ap_feature_importance)
    ap_outlier_subspace = qcml.AttachmentType(name='Outlier subspace', ID='{}_Subspace'.format(
        run_quality.get_ID()), qualityParameterRef=qp_score.get_ID(), binary=fig_subspace)
    run_quality.add_attachment(ap_outlier_subspace)

# TODO: style sheet for viewing in a browser

with open('out.qcml', 'w') as outfile:
    qcml_out.export(outfile, 0, name_='qcML', namespacedef_='xmlns="http://www.prime-xs.eu/ms/qcml"')


##############################################################################

# CLUSTERING

from snn_cluster import subspace_cluster


# print subspace_cluster(data.as_matrix(), 500)

# from sklearn.cluster import DBSCAN, KMeans
# from sklearn import metrics
#
# classes = []
# for filename in filenames:
#     classes.append(filename[5: 19])
#
#
# from sklearn.manifold import TSNE
# tsne_data = TSNE(2, 15, init='pca').fit_transform(data)
#
# NN distance plot
# from sklearn.neighbors import NearestNeighbors
# knn = NearestNeighbors(n_neighbors=10)
# knn.fit(data)
# distances = knn.kneighbors(data)
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure()
# sns.distplot(distances[0][:, -1])
# plt.show()
# plt.close()
#
# labels = DBSCAN(2, 15).fit_predict(tsne_data)
# print len(set(labels))
# # labels = KMeans(len(set(classes))).fit_predict(data)
#
# # print('Estimated number of clusters: %d' % n_clusters_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(classes, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(classes, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(classes, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(classes, labels))
# print("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(classes, labels))
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, labels))
#
# visualize.visualize_tsne(data, labels)
