import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from clusterValidation import valuate_kmeans
from dimensionReduction import applyPCA, applyFastICA, applyGRP, applyLDA
from sklearn.preprocessing import OneHotEncoder


training_data_file_name = "phishing_binary.txt"
label_file_name = "phishing_label.txt"
path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
data = np.loadtxt(open(path + training_data_file_name),  dtype = int, delimiter = ',')
labels = np.loadtxt(open(path + label_file_name),  dtype = int, delimiter = ',')


samples = len(data)
clusters = [2, 4, 8, 12, 16]

#new_data = applyPCA(data, 10, data)
#new_data = applyFastICA(data, 30, data)
#new_data = applyGRP(data, 30, data)
new_data = applyLDA(data, labels, 10, data)
new_features = np.zeros((samples, 5))
for i in range(5):
    fit_time, labels_pred = valuate_kmeans(new_data, clusters[i], new_data)
    new_features[:, i] = labels_pred
    new_features.astype(int)
    
    #Encode categorical integer features.
    enc = OneHotEncoder()
    enc.fit(new_features[:, :i+1])  
    new_binary_data = enc.transform(new_features[:, :i+1]).toarray()
    
    #np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\pca" + str(i+1) + "binary.txt", new_binary_data, fmt = "%i", delimiter = ",")
    #np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\ica" + str(i+1) + "binary.txt", new_binary_data, fmt = "%i", delimiter = ",")
    #np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\grp" + str(i+1) + "binary.txt", new_binary_data, fmt = "%i", delimiter = ",")
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\lda" + str(i+1) + "binary.txt", new_binary_data, fmt = "%i", delimiter = ",")
