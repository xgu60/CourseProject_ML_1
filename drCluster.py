import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from clusterValidation import valuate_kmeans
from dimensionReduction import applyPCA, applyFastICA, applyGRP, applyLDA

training_data_file_name = "phishing_binary.txt"
label_file_name = "phishing_label.txt"
path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
data = np.loadtxt(open(path + training_data_file_name),  dtype = int, delimiter = ',')
labels = np.loadtxt(open(path + label_file_name),  dtype = int, delimiter = ',')

clusters = 4
features = len(data[0])
sample_num = len(data)

ARS_pca = []
ARS_ica = []
ARS_grp = []
ARS_lda = []
components = []
for i in range(1, 10):
    components.append(i)
    
    pca_data = applyPCA(data, i, data)
    pca_time, labels_pca = valuate_kmeans(pca_data, clusters, pca_data) 
    ARS_pca.append(metrics.adjusted_rand_score(labels_pca, labels) )
    
    ica_data = applyFastICA(data, i, data)
    ica_time, labels_ica = valuate_kmeans(ica_data, clusters, ica_data) 
    ARS_ica.append(metrics.adjusted_rand_score(labels_ica, labels) )
    
    grp_data = applyGRP(data, i, data)
    grp_time, labels_grp = valuate_kmeans(grp_data, clusters, grp_data) 
    ARS_grp.append(metrics.adjusted_rand_score(labels_grp, labels) )
    
    lda_data = applyLDA(data, labels, i, data)
    lda_time, labels_lda = valuate_kmeans(lda_data, clusters, lda_data) 
    ARS_lda.append(metrics.adjusted_rand_score(labels_lda, labels) )
    
fit_time, labels_pred = valuate_kmeans(data, clusters, data) 
ARS_ori = metrics.adjusted_rand_score(labels_pred, labels) 
    
#print ARS_pca, '\n', ARS_ica, '\n', ARS_grp, '\n', ARS_lda
    
#plot samples using new components
plt.figure()
plt.plot(components, ARS_pca, c='r', linewidth=3, label='PCA')
plt.plot(components, ARS_ica, c='b', linewidth=3, label='ICA')
plt.plot(components, ARS_grp, c='c', linewidth=3, label='GRP')
plt.plot(components, ARS_lda, c='g', linewidth=3, label='LDA')
plt.scatter(features, ARS_ori, s=100, c='k', label='original')

plt.title("components vs ARS")
plt.xlabel("components")
plt.ylabel("ARS")   
#plt.ylim((-0.2, 1.2))
plt.legend(loc=(0.9, 0.75))
plt.show()    