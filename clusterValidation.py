import numpy as np
import time
import os
from sklearn.cluster import KMeans 
from sklearn import mixture
from sklearn import metrics
import matplotlib.pyplot as plt

#kmeans 
def valuate_kmeans(data, nc, new_data):
    kmeans = KMeans(n_clusters=nc, random_state=79)

    start_time = time.time()
    kmeans.fit(data)
    fit_time = time.time() - start_time

    return fit_time, kmeans.predict(new_data)

#Expectation Maximization using GaussianMixture
def valuate_GM(data, nc, new_data):
    gmm = mixture.GaussianMixture(n_components=nc, covariance_type='tied', max_iter=100, random_state=79)
    start_time = time.time()
    gmm.fit(data)
    fit_time = time.time() - start_time

    labels_pred = gmm.predict(new_data)
    #print gmm.converged_
    return fit_time, labels_pred

def main():
    #get data from text file
    training_data_file_name = "mushroom_binary.txt"
    label_file_name = "mushroom_label.txt"
    path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
    training_data = np.loadtxt(open(path + training_data_file_name),  dtype = int, delimiter = ',')
    labels_true = np.loadtxt(open(path + label_file_name),  dtype = int)

    clusters = []
    running_time = []
    ars = []
    amis = []
    hs = []
    cs = []
    vms = []
    
    for i in range(2, 50):
        clusters.append(i)
        fit_time, labels_pred = valuate_kmeans(training_data,i, training_data)
        running_time.append(fit_time)
        ars.append(metrics.adjusted_rand_score(labels_pred, labels_true))
        amis.append(metrics.adjusted_mutual_info_score(labels_pred, labels_true))
        hs.append(metrics.homogeneity_score(labels_pred, labels_true))
        cs.append(metrics.completeness_score(labels_pred, labels_true))
        vms.append(metrics.v_measure_score(labels_pred, labels_true))

    #plot samples using new components
    plt.figure()
    plt.plot(clusters, ars, c='r', linewidth=3, label='ARS')
    plt.plot(clusters, amis, c='b', linewidth=3, label='AMIS')
    plt.plot(clusters, hs, c='c', linewidth=3, label='HS')
    plt.plot(clusters, cs, c='g', linewidth=3, label='CS')
    plt.plot(clusters, vms, c='k', linewidth=3, label='VMS')
    plt.title("clusters vs scores")
    plt.xlabel("cluster number")
    plt.ylabel("scores")   
    plt.legend(loc=(1))
    plt.show()  
    
    plt.figure()
    plt.plot(clusters, running_time, c='k', linewidth=3, label='running time')
    plt.title("clusters vs running time")
    plt.xlabel("cluster number")
    plt.ylabel("time (second)")   
    plt.legend(loc=(2))
    plt.show()    
    

if __name__ == "__main__":
    main()