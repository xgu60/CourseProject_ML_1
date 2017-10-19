import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from clusterValidation import valuate_kmeans, valuate_GM
from dimensionReduction import applyPCA, applyFastICA, applyGRP, applyLDA

training_data_file_name = "phishing_binary.txt"
label_file_name = "phishing_label.txt"
path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
data = np.loadtxt(open(path + training_data_file_name),  dtype = int, delimiter = ',')
labels = np.loadtxt(open(path + label_file_name),  dtype = int, delimiter = ',')

clusters = 4
features = len(data[0])
sample_num = len(data)

ARS = []
VMS = []
ARS2 = []
VMS2 = []

components = []
for i in range(1, 40):
    components.append(i)
        
    grp_data = applyGRP(data, i)
    grp_time, labels_grp = valuate_kmeans(grp_data, clusters) 
    grp_time2, labels_grp2 = valuate_GM(grp_data, clusters)
    ARS.append(metrics.adjusted_rand_score(labels_grp, labels) )
    VMS.append(metrics.v_measure_score(labels_grp, labels) )
    ARS2.append(metrics.adjusted_rand_score(labels_grp2, labels) )
    VMS2.append(metrics.v_measure_score(labels_grp2, labels) )
    
    
   
#print ARS_pca, '\n', ARS_ica, '\n', ARS_grp, '\n', ARS_lda
    
#plot samples using new components
plt.figure()
plt.plot(components, ARS, c='r', linewidth=1, label='ARS-GRP')
plt.plot(components, ARS2, c='r', linestyle=':', linewidth=3, label='ARS-ori')
plt.plot(components, VMS, c='b', linewidth=1, label='VMS-GRP')
plt.plot(components, VMS2, c='b', linestyle=':', linewidth=3, label='VMS-ori')

plt.title("clustering performance vs components")
plt.xlabel("components")
plt.ylabel("scores")   
#plt.ylim((-0.2, 1.2))
plt.legend(loc=(2))
plt.show()    