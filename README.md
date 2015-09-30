Analysis of mass spectrometry quality control metrics
=====================================================

For more information:

* [Official website](https://bitbucket.org/proteinspector/qc_outlier/)

This tool analyzes the quality of mass spectrometry experiments based on their quality control metrics.

Application
-----------

    usage: qc_analysis.py [-h] [--min_var MIN_VAR] [--min_corr MIN_CORR]
                          [--scaling_mode {robust,standard}] --k_neighbors
                          K_NEIGHBORS [--distance DISTANCE]
                          [--min_outlier MIN_OUTLIER] [--num_bins NUM_BINS]
                          [--min_sup MIN_SUP] [--min_length MIN_LENGTH]
                          file_in file_out

    Mass spectrometry quality control metrics analysis

    positional arguments:
      file_in               the tab-separated input file containing the QC metrics
      file_out              the name of the qcML output file

    optional arguments:
      -h, --help            show this help message and exit
      --min_var MIN_VAR, -var MIN_VAR
                            metrics with a lower variance will be removed
                            (default: 0.0001)
      --min_corr MIN_CORR, -corr MIN_CORR
                            metrics with a higher correlation will be removed
                            (default: 0.9)
      --scaling_mode {robust,standard}, -scale {robust,standard}
                            mode to standardize the metric values (default:
                            robust)
      --k_neighbors K_NEIGHBORS, -k K_NEIGHBORS
                            the number of nearest neighbors used for outlier
                            detection
      --distance DISTANCE, -dist DISTANCE
                            metric to use for distance computation (default:
                            manhattan) any metric from scikit-learn or
                            scipy.spatial.distance can be used
      --min_outlier MIN_OUTLIER, -o MIN_OUTLIER
                            the minimum outlier score threshold (default: None) if
                            no threshold is provided, an automatic threshold is
                            determined
      --num_bins NUM_BINS, -bin NUM_BINS
                            the number of bins for the outlier score histogram
                            (default: 20)
      --min_sup MIN_SUP, -sup MIN_SUP
                            the minimum support for subspace frequent itemset
                            mining (default: 5) positive numbers are interpreted
                            as percentages, negative numbers as absolute supports
      --min_length MIN_LENGTH, -len MIN_LENGTH
                            the minimum length each subspace itemset should be
                            (default: 1)

The only **required** parameters are the QC metrics input file, the qcML output file, and the number of neighbors used for detecting outlying experiments (--k_neighbors / -k).

The other parameters are optional and can be used to optimize the analyses, however, the default values should function adequately in most situations.

Input file format
-----------------

The input file must be a tab-separated file containing the QC metrics, and must adhere to the following format:

	Filename		StartTimeStamp		Metric0		Metric1		...		MetricN
	filename0		date0				value		value		...		value
	filename1		date1				value		value		...		value
	...

The first row contains the headers, the subsequent rows each contain the metrics for a single experiment. Columns are separated by tabs and the column values should not contain spaces.

The first two columns containing the filename and the experiment date are **mandatory**, and the filenames should be unique. Also, the headers for these two columns need to be equal to 'Filename' and 'StartTimeStamp'.

Subsequent columns specify the values for the various metrics. All metrics should have only **numeric** values. There is no restriction on the number of metrics columns or the denomination of the metrics column headers.

Files adhering to this format can be generated directly using [QuaMeter](http://pubs.acs.org/doi/abs/10.1021/ac300629p).

Output file format
------------------

The result of the analysis is exported to a qcML file, which can be viewed in any browser.

Dependencies
------------

Several Python libraries are required, with their minimal version garantueed to work as listed below:

* [lxml](http://lxml.de/) 3.4.4
* [matplotlib](http://matplotlib.org/) 1.4.3
* [NumPy](http://www.numpy.org/) 1.9.2
* [pandas](http://pandas.pydata.org/) 0.16.0
* [PyFIM](http://www.borgelt.net/pyfim.html) 6.12
* [Seaborn](https://web.stanford.edu/~mwaskom/software/seaborn/) 0.6.0
* [scikit-learn](http://scikit-learn.org/) 0.17.dev
* [SciPy](https://www.scipy.org/) 0.15.1
* [sklearn-pandas](https://github.com/paulgb/sklearn-pandas) 0.0.10

Contact
-------

For more information you can visit the [official code website](https://bitbucket.org/proteinspector/qc_analysis/) and send a message through Bitbucket or send a mail to <wout.bittremieux@uantwerpen.be>.
