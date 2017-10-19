import numpy as np
import os
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt



#dimension reduction using PCA
def applyPCA(data, nc, new_data):
    pca = PCA(n_components=nc, random_state=79)
    pca.fit(data)  
    return pca.transform(new_data)

             
#dimension reduction using FastICA
def applyFastICA(data, nc, new_data):
    ica = FastICA(n_components=nc, random_state=79)
    ica.fit(data)   
    return ica.transform(new_data)

#dimension reduction using Gaussian Radomized Projection
def applyGRP(data, nc, new_data):
    grp = GaussianRandomProjection(n_components=nc, random_state=79)
    grp.fit(data) 
    return grp.transform(new_data)

#dimension reduction using LinearDiscriminantAnalysis
def applyLDA(data, labels, nc, new_data):
    lda = LinearDiscriminantAnalysis(solver='eigen', n_components=nc, shrinkage='auto')
    lda.fit(data, labels)
    return lda.transform(new_data)

#plot 
def plotTwoD(data, data_labels, title, xlabel, ylabel, l1, l2):
    #separate data for plotting
    labelSample1 = []
    labelSample2 = []
    for i in range(len(data_labels)):
        if data_labels[i] == 1:
            labelSample1.append([data[i,0], data[i,1]])
        else:
            labelSample2.append([data[i,0], data[i,1]])
    labelSample1 = np.array(labelSample1)      
    labelSample2 = np.array(labelSample2) 
     
    #plot samples using new components
    plt.figure()
    plt.scatter(labelSample1[:, 0], labelSample1[:, 1], c='r', label=l1)
    plt.scatter(labelSample2[:, 0], labelSample2[:, 1], c='b', label=l2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)   
    plt.legend(loc=2)
    plt.show()


def main():                
    #get data from txt file
    training_data_file_name = "phishing_binary.txt"
    label_file_name = "phishing_label.txt"
    path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
    data = np.loadtxt(open(path + training_data_file_name), delimiter = ',')
    labels = np.loadtxt(open(path + label_file_name),  dtype = int, delimiter = ',')

    #new data after dimension reduction
    pca_data = applyPCA(data, 2, data)
    ica_data = applyFastICA(data, 2, data)
    grp_data = applyGRP(data, 2, data)
    lda_data = applyLDA(data, labels, 2, data)

    #plot 
    #plotTwoD(pca_data, labels, 'PCA analysis of mushroom dataset', 'first_component', 'second_component', 'poison', 'edible')
    #plotTwoD(ica_data, labels, 'ICA analysis of mushroom dataset', 'first_component', 'second_component', 'poison', 'edible')
    #plotTwoD(grp_data, labels, 'GPR analysis of mushroom dataset', 'first_component', 'second_component', 'poison', 'edible')
    #plotTwoD(lda_data, labels, 'LDA analysis of mushroom dataset', 'first_component', 'second_component', 'poison', 'edible')
    plotTwoD(pca_data, labels, 'PCA analysis of phishing websites dataset', 'first_component', 'second_component', 'phishing', 'normal')
    plotTwoD(ica_data, labels, 'ICA analysis of phishing websites dataset', 'first_component', 'second_component', 'phishing', 'normal')
    plotTwoD(grp_data, labels, 'GPR analysis of phishing websites dataset', 'first_component', 'second_component', 'phishing', 'normal')
    plotTwoD(lda_data, labels, 'LDA analysis of phishing websites dataset', 'first_component', 'second_component', 'phishing', 'normal')
    
if __name__ == "__main__":
    main()