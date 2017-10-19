import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from dimensionReduction import applyPCA, applyFastICA, applyGRP, applyLDA

#generate learning curve     
def learningCurveMLP(training_data, training_labels, testing_data, testing_labels):
    per = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sample_num = len(training_data)
    training_score = []
    testing_score = []
    for i in range(10):
        t_num = int(sample_num * per[i])
        nnl = MLPClassifier(max_iter=1000)
        nnl.fit(training_data[:t_num], training_labels[:t_num])

        training_score.append(nnl.score(training_data[:t_num], training_labels[:t_num]))
        testing_score.append(nnl.score(testing_data, testing_labels))

    return training_score, testing_score
    

training_data_file_name = "phishing_binary.txt"
label_file_name = "phishing_label.txt"
path = os.path.dirname(os.path.realpath(__file__)) + "\\data\\"
data = np.loadtxt(open(path + training_data_file_name),  delimiter = ',')
labels = np.loadtxt(open(path + label_file_name),  dtype = int, delimiter = ',')

sampleNum = len(data)
trainingNum = int(sampleNum * 0.7)

trainingCurve = []
testingCurve = []
#original training, and testing curve
ori_training_score, ori_testing_score = learningCurveMLP(data[:trainingNum], labels[:trainingNum], data[trainingNum:], labels[trainingNum:])
trainingCurve.append(ori_training_score)
testingCurve.append(ori_testing_score)

nc = [2, 4, 8, 16, 32, 64]

#training and testing curve after dimension reduction using PCA
for i in range(6):
    training_data = applyPCA(data[:trainingNum], nc[i], data[:trainingNum])
    testing_data = applyPCA(data[:trainingNum], nc[i], data[trainingNum :])
    training_score, testing_score = learningCurveMLP(training_data, labels[:trainingNum], testing_data, labels[trainingNum:])
    trainingCurve.append(training_score)
    testingCurve.append(testing_score)
    
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "pca_training.txt", trainingCurve, fmt = "%f", delimiter = ",")  
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "pca_testing.txt", testingCurve, fmt = "%f", delimiter = ",")   
'''
#training and testing curve after dimension reduction using ICA
for i in range(6):
    training_data = applyFastICA(data[:trainingNum], nc[i], data[:trainingNum])
    testing_data = applyFastICA(data[:trainingNum], nc[i], data[trainingNum :])
    training_score, testing_score = learningCurveMLP(training_data, labels[:trainingNum], testing_data, labels[trainingNum:])
    trainingCurve.append(training_score)
    testingCurve.append(testing_score)
    
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "ica_training.txt", trainingCurve, fmt = "%f", delimiter = ",")  
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "ica_testing.txt", testingCurve, fmt = "%f", delimiter = ",")   

#training and testing curve after dimension reduction using ICA
for i in range(6):
    training_data = applyGRP(data[:trainingNum], nc[i], data[:trainingNum])
    testing_data = applyGRP(data[:trainingNum], nc[i], data[trainingNum :])
    training_score, testing_score = learningCurveMLP(training_data, labels[:trainingNum], testing_data, labels[trainingNum:])
    trainingCurve.append(training_score)
    testingCurve.append(testing_score)
    
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "grp_training.txt", trainingCurve, fmt = "%f", delimiter = ",")  
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "grp_testing.txt", testingCurve, fmt = "%f", delimiter = ",")   

#training and testing curve after dimension reduction using ICA
for i in range(6):
    training_data = applyLDA(data[:trainingNum], labels[:trainingNum], nc[i], data[:trainingNum])
    testing_data = applyLDA(data[:trainingNum], labels[:trainingNum], nc[i], data[trainingNum:])
    training_score, testing_score = learningCurveMLP(training_data, labels[:trainingNum], testing_data, labels[trainingNum:])
    trainingCurve.append(training_score)
    testingCurve.append(testing_score)
    
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "lda_training.txt", trainingCurve, fmt = "%f", delimiter = ",")  
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\" + "lda_testing.txt", testingCurve, fmt = "%f", delimiter = ",")   
'''