import qcml
import datetime
import numpy as np
import pandas as pd

import visualize


class QcmlExport:
    qcml_out = qcml.qcMLType()
    set_quality = qcml.SetQualityAssessmentType(ID='OutlierAnalysis')

    cv_ms = qcml.CVType('PSI-MS', '3.78.0', 'http://psidev.cvs.sourceforge.net/viewvc/psidev/psi/psi-ms/mzML/controlledVocabulary/psi-ms.obo', 'MS')
    cv_qc = qcml.CVType('MS-QC', '0.1.1', 'https://github.com/qcML/qcML-development/blob/master/cv/qc-cv.obo', 'QC')
    cv_quameter = qcml.CVType('QuaMeter', '0', 'http://pubs.acs.org/doi/abs/10.1021/ac300629p', 'QM')

    creation_date = datetime.datetime.now()

    def __init__(self):
        self.qcml_out.set_version('0.0.8')

        # add references to the controlled vocabularies (CV's)
        self.qcml_out.set_cvList(qcml.CVListType([self.cv_ms, self.cv_qc, self.cv_quameter]))

        # add global outlier information in a setQuality
        self.qcml_out.add_setQuality(self.set_quality)

    def add_low_variance(self, variances, min_var):
        param_var = qcml.QualityParameterType(name='Variance threshold', ID='VarianceThreshold', value='{:.3e}'.format(min_var))
        self.set_quality.add_qualityParameter(param_var)

        values = [' '.join((v[0], '{:.3e}'.format(v[1]))) for v in variances[variances <= min_var].iteritems()]
        table = qcml.TableType(tableColumnTypes=['Metric', 'Variance'], tableRowValues=values)
        self.set_quality.add_attachment(qcml.AttachmentType(name='Low variance metrics', ID='var', table=table, qualityParameterRef=param_var.get_ID()))

    def add_correlation(self, corr, min_corr):
        param_corr = qcml.QualityParameterType(name='Correlation threshold', ID='CorrelationThreshold', value=min_corr)
        self.set_quality.add_qualityParameter(param_corr)

        values = []
        corr_features = set()
        for row in range(len(corr.index)):
            if corr.columns.values[row] not in corr_features:
                for col in range(row + 1, len(corr.columns)):
                    if corr.columns.values[col] not in corr_features and abs(corr.iloc[row, col]) > min_corr:
                        corr_features.add(corr.columns.values[col])
                        values.append(' '.join((corr.columns.values[row], corr.columns.values[col], '{:.2%}'.format(corr.iloc[row, col]))))
        table = qcml.TableType(tableColumnTypes=['Metric', 'Metric', 'Correlation'], tableRowValues=values)
        self.set_quality.add_attachment(qcml.AttachmentType(name='Correlated metrics', ID='corr', table=table, qualityParameterRef=param_corr.get_ID()))

    def add_visualization(self, data):
        self.set_quality.add_attachment(qcml.AttachmentType(name='Experiment execution time', ID='time',
                                                            binary=visualize.visualize_timestamps(data, filename='__qcml_export__')))
        self.set_quality.add_attachment(qcml.AttachmentType(name='PCA visualization', ID='PCA',
                                                            binary=visualize.visualize_pca(data, filename='__qcml_export__')))
        self.set_quality.add_attachment(qcml.AttachmentType(name='t-SNE visualization', ID='t-SNE',
                                                            binary=visualize.visualize_tsne(data, filename='__qcml_export__')))

    def add_outlier_scores(self, outlier_scores, outlier_threshold):
        param_score = qcml.QualityParameterType(name='Outlier score threshold', ID='OutlierScoreThreshold', value=outlier_threshold)
        self.set_quality.add_qualityParameter(param_score)

        attach_hist = qcml.AttachmentType(name='Outlier score histogram', ID='OutlierScoreHistogram', qualityParameterRef=param_score.get_ID(),
                                          binary=visualize.plot_outlier_score_hist(outlier_scores, outlier_threshold, filename='__qcml_export__'))
        self.set_quality.add_attachment(attach_hist)

    def add_outlier_runquality(self, outlier, data):
        run_quality = qcml.RunQualityAssessmentType(ID=outlier.name[0])
        self.qcml_out.add_runQuality(run_quality)

        run_quality.add_metaDataParameter(qcml.MetaDataType(name='Creation date', value=self.creation_date,
                                                            cvRef=self.cv_qc.get_ID(), accession='MS:1000747',
                                                            ID='{}_CreationDate'.format(run_quality.get_ID())))

        score = qcml.QualityParameterType(name='Outlier score', value=outlier['OutlierScore'],
                                          ID='{}_OutlierScore'.format(run_quality.get_ID()))
        run_quality.add_qualityParameter(score)

        feature_importance = pd.Series(outlier['FeatureImportance'], index=outlier.drop(['OutlierScore', 'FeatureImportance', 'Subspace']).index)

        fig_features = visualize.visualize_feature_importances(feature_importance, filename='__qcml_export__')
        fig_subspace = visualize.visualize_subspace_boxplots(data[outlier['Subspace']], outlier[outlier['Subspace']], filename='__qcml_export__')

        run_quality.add_attachment(qcml.AttachmentType(name='Feature importance', binary=fig_features,
                                   ID='{}_FeatureImportance'.format(run_quality.get_ID()),
                                   qualityParameterRef=score.get_ID()))
        run_quality.add_attachment(qcml.AttachmentType(name='Outlier subspace', binary=fig_subspace,
                                                       ID='{}_Subspace'.format(run_quality.get_ID()),
                                                       qualityParameterRef=score.get_ID()))

    def export(self, filename):
        # TODO: style sheet for viewing in a browser
        with open(filename, 'w') as outfile:
            self.qcml_out.export(outfile, 0, name_='qcML', namespacedef_='xmlns="http://www.prime-xs.eu/ms/qcml"')
