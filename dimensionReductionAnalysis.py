import numpy as np
import os
from scipy.stats import kurtosis
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt




#dimension reduction using PCA

def PCA_analysis(data, nc):
    pca = PCA(n_components=nc, random_state=79)
    pca.fit(data)  
    return pca.explained_variance_ratio_
             
#dimension reduction using FastICA
def FastICA_analysis(data, nc):
    ica = FastICA(n_components=nc, random_state=79)
    ica.fit(data)
    return kurtosis(ica.mixing_, axis=0, fisher=True, bias=True)
       

 

    
#dimension reduction using LinearDiscriminantAnalysis
def LDA_analysis(data, labels, nc):
    lda = LinearDiscriminantAnalysis(solver='eigen', n_components=nc, shrinkage='auto')
    lda.fit(data, labels)
    var = np.var(lda.transform(data), axis=0)
    return var / np.sum(var)
    
    
def main():                
    #get data from txt file
    data_file_name1 = "phishing_binary.txt"
    data_file_name2 = "mushroom_binary.txt"
    label_file_name1 = "phishing_label.txt"
    label_file_name2 = "mushroom_label.txt"
    path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
    data1 = np.loadtxt(open(path + data_file_name1), delimiter = ',')
    data2 = np.loadtxt(open(path + data_file_name2), delimiter = ',')
    labels1 = np.loadtxt(open(path + label_file_name1),  dtype = int, delimiter = ',')
    labels2 = np.loadtxt(open(path + label_file_name2),  dtype = int, delimiter = ',')

    #PCA
    #phishing = PCA_analysis(data1, 20)
    #mushroom = PCA_analysis(data2, 20)
    
    #FastICA
    #mk1 = []
    #mk2 = []
    #for i in range(1, 30):
    #    phishing = FastICA_analysis(data1, i)
    #    mushroom = FastICA_analysis(data2, i)
        
    #    mk1.append(max(phishing))
    #    mk2.append(max(mushroom))
    
    #LDA
    phishing = LDA_analysis(data1, labels1, 10)
    mushroom = LDA_analysis(data2, labels2, 10)
    
    nc = np.array(range(1, 11))
    
    
    #plot         
    plt.figure()
    plt.plot(nc, phishing, c='r', linewidth=2, label='phishing')
    plt.plot(nc, mushroom, c='b', linewidth=2, label='mushroom')
    
    plt.title('explained_variance_ratio vs components', fontsize=18)
    plt.xlabel('components', fontsize=14)
    plt.ylabel('variance_ratio', fontsize=14)   
    plt.legend(loc=1)
    plt.show()
    
    
if __name__ == "__main__":
    main()