import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def toNumericalLabels(startFileName, endFileName):
    data = np.loadtxt(open(os.path.dirname(os.path.realpath(__file__)) + "\\data\\" + startFileName), dtype = str, delimiter = ',')
    num_data = np.ones_like(data, dtype = int)

    for i in range(data.shape[1]):
        le = LabelEncoder()
        le.fit(data[:, i])
    
        num_data[:, i] = le.transform(data[:,i])

 
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\data\\" + endFileName, num_data, fmt = "%i", delimiter = ",")

def toBinaryFeature(startFileName, endFileName, labelFileName, indexOfLabel):
    path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
    data = np.loadtxt(open(path + startFileName),  dtype = int, delimiter = ',')
    cols = len(data[0])

    training_data = np.delete(data, indexOfLabel, 1)
    labels_true = data[:, indexOfLabel]

    #Encode categorical integer features.
    enc = OneHotEncoder()
    enc.fit(training_data)  
    new_training_data = enc.transform(training_data).toarray()
    
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\data\\" + endFileName, new_training_data, fmt = "%i", delimiter = ",")
    np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\data\\" + labelFileName, labels_true, fmt = "%i", delimiter = ",")

def main():
    
    toNumericalLabels("mushroom.txt", "mushroom_num.txt")
    toNumericalLabels("phishing.txt", "phishing_num.txt")
    
    toBinaryFeature("mushroom_num.txt", "mushroom_binary.txt", "mushroom_label.txt", 0)
    toBinaryFeature("phishing_num.txt", "phishing_binary.txt", "phishing_label.txt", -1)

if __name__ == "__main__":
    main()