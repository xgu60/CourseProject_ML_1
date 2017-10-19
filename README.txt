In this fold contains:

Three folds:
1, data fold (containing original data and the data after preprocess)
2, learningCurve fold (temporary store learning curve data for plotting)
3, new_features fold (contains txt files to store new genearted datasets used in part 5)

Ten python files:
dataPreprocess: used to change catagory features into binary features, as shown in Figure 1
clusterValidation: use k-means and gaussianmixture to cluster datasets, and used to generate Figure 2
dimensionReduction: apply PCA, FastICA, GRP and LDA to datasets, and used to generate Figure 3
dimensionReductionAnalysis, grp_analysis: used to analyze new generated components, and used to generate Figure 4
drCluster: run clustering on dimensional reduced dataset, used to generate Figure 5
NNLearner: apply neural network on dimensional reduced datasets, store data into temp text files, used to generate Figure 6
generateNewFeatures: generate new datasets by using clustering algorithm on dimensional reduced datasets.
newFeatureLC: apply neural network on new feature datasets and store data into temp text file for plotting, used to generate Figure 7
lcPlot: plot data points from text file, used to plot Figure 6 and 7.

README file

xgu60-analysis.pdf