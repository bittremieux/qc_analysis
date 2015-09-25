import datetime
import os

import pandas as pd
from sklearn.decomposition import PCA
from sklearn_pandas import DataFrameMapper

import qcml
import visualize


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


class Exporter:

    def __init__(self, export_qcml=True, export_figures=False):
        self.export_qcml = export_qcml
        self.export_figures = export_figures

        self.creation_date = datetime.datetime.now()

        if self.export_qcml:
            # create qcML
            self.qcml_out = qcml.qcMLType()
            self.qcml_out.set_version('0.0.8')

            # add references to the controlled vocabularies (CV's)
            self.cv_ms = qcml.CVType('PSI-MS', '3.78.0', 'http://psidev.cvs.sourceforge.net/viewvc/psidev/psi/psi-ms/mzML/controlledVocabulary/psi-ms.obo', 'MS')
            self.cv_qc = qcml.CVType('MS-QC', '0.1.1', 'https://github.com/qcML/qcML-development/blob/master/cv/qc-cv.obo', 'QC')
            self.cv_outlier = qcml.CVType('Outlier detection and interpretation', '0', 'article/code link', 'outlier')

            self.qcml_out.set_cvList(qcml.CVListType([self.cv_ms, self.cv_qc, self.cv_outlier]))

            # add global outlier information in a setQuality
            self.set_quality = qcml.SetQualityAssessmentType(ID='OutlierAnalysis')
            self.set_quality.add_metaDataParameter(qcml.MetaDataType(name='Creation date', value=self.creation_date,
                                                                     cvRef=self.cv_qc.get_ID(), accession='MS:1000747',
                                                                     ID='{}_CreationDate'.format(
                                                                         self.set_quality.get_ID())))
            self.qcml_out.add_setQuality(self.set_quality)

            # add embedded stylesheet
            with open('to_html.xsl', 'r') as xsl:
                self.qcml_out.set_embeddedStylesheetList(qcml.embeddedStylesheetListType(anytypeobjs_=xsl.read()))

        if self.export_figures:
            pass

    def low_variance(self, variances, min_var):
        if self.export_qcml:
            param_var = qcml.QualityParameterType(name='Variance threshold', ID='VarianceThreshold', value='{:.3e}'.format(min_var),
                                                  cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_qualityParameter(param_var)

            values = [' '.join((v[0], '{:.3e}'.format(v[1]))) for v in variances[variances <= min_var].iteritems()]
            table = qcml.TableType(tableColumnTypes=['RemovedMetric', 'Variance'], tableRowValues=values)
            self.set_quality.add_attachment(qcml.AttachmentType(name='Low variance metrics', ID='var', table=table,
                                                                cvRef=self.cv_outlier.get_ID(), accession='none',
                                                                qualityParameterRef=param_var.get_ID()))
        if self.export_figures:
            pass

    def correlation(self, corr, min_corr):
        if self.export_qcml:
            param_corr = qcml.QualityParameterType(name='Correlation threshold', ID='CorrelationThreshold', value=min_corr,
                                                   cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_qualityParameter(param_corr)

            values = []
            corr_features = set()
            for row in range(len(corr.index)):
                if corr.columns.values[row] not in corr_features:
                    for col in range(row + 1, len(corr.columns)):
                        if corr.columns.values[col] not in corr_features and abs(corr.iloc[row, col]) > min_corr:
                            corr_features.add(corr.columns.values[col])
                            values.append(' '.join((corr.columns.values[row], corr.columns.values[col], '{:.2%}'.format(corr.iloc[row, col]))))
            table = qcml.TableType(tableColumnTypes=['RetainedMetric', 'RemovedMetric', 'Correlation'], tableRowValues=values)
            self.set_quality.add_attachment(qcml.AttachmentType(name='Correlated metrics', ID='corr', table=table,
                                                                cvRef=self.cv_outlier.get_ID(), accession='none',
                                                                qualityParameterRef=param_corr.get_ID()))

        if self.export_figures:
            visualize.visualize_correlation_matrix(corr, 'corr.pdf')

    def global_visualization(self, data):
        if self.export_qcml:
            self.set_quality.add_attachment(qcml.AttachmentType(name='Experiment execution time', ID='time',
                                                                binary=visualize.visualize_timestamps(data, filename='__qcml_export__'),
                                                                cvRef=self.cv_outlier.get_ID(), accession='none'))
            self.set_quality.add_attachment(qcml.AttachmentType(name='PCA visualization', ID='PCA',
                                                                binary=visualize.visualize_pca(data, filename='__qcml_export__'),
                                                                cvRef=self.cv_outlier.get_ID(), accession='none'))
            self.set_quality.add_attachment(qcml.AttachmentType(name='t-SNE visualization', ID='t-SNE',
                                                                binary=visualize.visualize_tsne(data, filename='__qcml_export__'),
                                                                cvRef=self.cv_outlier.get_ID(), accession='none'))

        if self.export_figures:
            visualize.visualize_timestamps(data, filename='dates.pdf')
            visualize.visualize_pca(data, filename='pca.pdf')
            visualize.visualize_tsne(data, filename='tsne.pdf')

            pca = PCA(2)
            DataFrameMapper([(data.columns.values, pca)]).fit_transform(data)
            with open('table_pca.txt', 'w') as f_out:
                f_out.write(pca_loadings_table(pca, data.columns.values))

    def outlier_scores(self, outlier_scores, outlier_threshold, num_bins):
        if self.export_qcml:
            param_score = qcml.QualityParameterType(name='Outlier score threshold', ID='OutlierScoreThreshold', value=outlier_threshold,
                                                    cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_qualityParameter(param_score)
            param_nr = qcml.QualityParameterType(name='Number of outliers', ID='NrOutliers', value=(outlier_scores > outlier_threshold).sum(),
                                                 cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_qualityParameter(param_nr)

            attach_hist = qcml.AttachmentType(name='Outlier score histogram', ID='OutlierScoreHistogram', qualityParameterRef=param_score.get_ID(),
                                              binary=visualize.plot_outlier_score_hist(outlier_scores, num_bins, outlier_threshold, filename='__qcml_export__'),
                                              cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_attachment(attach_hist)

        if self.export_figures:
            visualize.plot_outlier_score_hist(outlier_scores, num_bins, outlier_threshold, filename='outlier-hist.pdf')

    def outlier(self, outlier, data):
        feature_importance = pd.Series(outlier['FeatureImportance'], index=outlier.drop(['OutlierScore', 'FeatureImportance', 'Subspace']).index)

        if self.export_qcml:
            run_quality = qcml.RunQualityAssessmentType(ID=outlier.name[0])
            self.qcml_out.add_runQuality(run_quality)

            run_quality.add_metaDataParameter(qcml.MetaDataType(name='Creation date', value=self.creation_date,
                                                                cvRef=self.cv_qc.get_ID(), accession='MS:1000747',
                                                                ID='{}_CreationDate'.format(run_quality.get_ID())))

            score = qcml.QualityParameterType(name='Outlier score', value=outlier['OutlierScore'],
                                              ID='{}_OutlierScore'.format(run_quality.get_ID()),
                                              cvRef=self.cv_outlier.get_ID(), accession='none')
            run_quality.add_qualityParameter(score)

            fig_features = visualize.visualize_feature_importances(feature_importance, filename='__qcml_export__')
            fig_subspace = visualize.visualize_subspace_boxplots(data[outlier['Subspace']], outlier[outlier['Subspace']], filename='__qcml_export__')

            run_quality.add_attachment(qcml.AttachmentType(name='Feature importance', binary=fig_features,
                                                           ID='{}_FeatureImportance'.format(run_quality.get_ID()),
                                                           cvRef=self.cv_outlier.get_ID(), accession='none',
                                                           qualityParameterRef=score.get_ID()))
            run_quality.add_attachment(qcml.AttachmentType(name='Explanatory subspace', binary=fig_subspace,
                                                           ID='{}_Subspace'.format(run_quality.get_ID()),
                                                           cvRef=self.cv_outlier.get_ID(), accession='none',
                                                           qualityParameterRef=score.get_ID()))

        if self.export_figures:
            if not os.path.exists('./outlier/'):
                os.makedirs('./outlier/')
            visualize.visualize_feature_importances(feature_importance, filename='./outlier/{}_features.pdf'.format(outlier.name[0]))
            visualize.visualize_subspace_boxplots(data[outlier['Subspace']], outlier[outlier['Subspace']], filename='./outlier/{}_subspace.pdf'.format(outlier.name[0]))

    def frequent_outlier_subspaces(self, subspaces, min_sup, min_length):
        if self.export_qcml:
            param_support = qcml.QualityParameterType(name='Minimum support', ID='minsup',
                                                      value='{}{}'.format(min_sup, '%' if min_sup > 0 else -1 * min_sup, ''),
                                                      cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_qualityParameter(param_support)
            param_length = qcml.QualityParameterType(name='Minimum subspace length', ID='minlength', value=min_length,
                                                     cvRef=self.cv_outlier.get_ID(), accession='none')
            self.set_quality.add_qualityParameter(param_length)

            values = ['{} {}'.format(subspace.iloc[0].replace(', ', '_'), subspace.iloc[1]) for _, subspace in subspaces.iterrows()]
            table = qcml.TableType(tableColumnTypes=['Subspace', 'NrOutliers'], tableRowValues=values)
            self.set_quality.add_attachment(qcml.AttachmentType(name='Frequently occuring explanatory subspaces', ID='freq',
                                                                table=table, qualityParameterRef=param_support.get_ID(),
                                                                cvRef=self.cv_outlier.get_ID(), accession='none'))

        if self.export_figures:
            with open('table_freq.txt', 'w') as f_out:
                f_out.write(subspaces.to_latex(index=False))

    def export(self, filename):
        if self.export_qcml:
            with open(filename, 'w') as outfile:
                self.qcml_out.export(outfile, 0, name_='qcML', namespacedef_='xmlns="http://www.prime-xs.eu/ms/qcml"')

        if self.export_figures:
            pass
