import os
import numpy as np
from NNLearner import learningCurveMLP

data_file = "phishing_binary.txt"
label_file = "phishing_label.txt"

path = os.path.dirname(os.path.realpath(__file__)) 

data = np.loadtxt(open(path + "\\data\\" + data_file),  delimiter = ',')
labels = np.loadtxt(open(path + "\\data\\" + label_file),  dtype = int, delimiter = ',')

sampleNum = len(data)
trainingNum = int(sampleNum * 0.7)

trainingCurve = []
testingCurve = []

#original training, and testing curve
ori_training_score, ori_testing_score = learningCurveMLP(data[:trainingNum], labels[:trainingNum], data[trainingNum:], labels[trainingNum:])
trainingCurve.append(ori_training_score)
testingCurve.append(ori_testing_score)

for i in range(5):
    #new_data_file = "pca" + str(i+1) + "binary.txt"
    #new_data_file = "ica" + str(i+1) + "binary.txt"
    #new_data_file = "grp" + str(i+1) + "binary.txt"
    new_data_file = "lda" + str(i+1) + "binary.txt"
    new_data = np.loadtxt(open(path + "\\new_features\\" + new_data_file),  delimiter = ',')
    training_score, testing_score = learningCurveMLP(new_data[:trainingNum], labels[:trainingNum], new_data[trainingNum:], labels[trainingNum:])
    trainingCurve.append(training_score)
    testingCurve.append(testing_score)

#np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "pcaNewFeatureTraining.txt", trainingCurve, fmt = "%f", delimiter = ",")  
#np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "pcaNewFeatureTesting.txt", testingCurve, fmt = "%f", delimiter = ",")  
#np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "icaNewFeatureTraining.txt", trainingCurve, fmt = "%f", delimiter = ",")  
#np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "icaNewFeatureTesting.txt", testingCurve, fmt = "%f", delimiter = ",")       
#np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "grpNewFeatureTraining.txt", trainingCurve, fmt = "%f", delimiter = ",")  
#np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "grpNewFeatureTesting.txt", testingCurve, fmt = "%f", delimiter = ",")       
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "ldaNewFeatureTraining.txt", trainingCurve, fmt = "%f", delimiter = ",")  
np.savetxt(os.path.dirname(os.path.realpath(__file__)) + "\\new_features\\" + "ldaNewFeatureTesting.txt", testingCurve, fmt = "%f", delimiter = ",")       


