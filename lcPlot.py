import numpy as np
import os
import matplotlib.pyplot as plt

trainingCurve = np.loadtxt(open(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\lda_training.txt"), delimiter = ',')
testingCurve = np.loadtxt(open(os.path.dirname(os.path.realpath(__file__)) + "\\learningCurve\\lda_testing.txt"), delimiter = ',')

x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
#plot training curve
plt.figure()
plt.plot(x, trainingCurve[0], c='k', linewidth=4, label='original')
plt.plot(x, trainingCurve[1], c='b', linewidth=2, label='2')
plt.plot(x, trainingCurve[2], c='b', linestyle = '--', linewidth=3, label='4')
plt.plot(x, trainingCurve[3], c='g', linewidth=2, label='8')
plt.plot(x, trainingCurve[4], c='g', linestyle = '--', linewidth=3, label='12')
plt.plot(x, trainingCurve[5], c='r', linewidth=2, label='16')
plt.plot(x, trainingCurve[6], c='r', linewidth=2, label='32')

plt.title("training curve using LDA ")
plt.xlabel("percentage %")
plt.ylabel("accuracy")   
plt.ylim((0,1.05))
plt.legend(loc=(4))
plt.show()    

#plot testing curve
plt.figure()
plt.plot(x, testingCurve[0], c='k', linewidth=4, label='original')
plt.plot(x, testingCurve[1], c='b', linewidth=2, label='2')
plt.plot(x, testingCurve[2], c='b', linestyle = '--', linewidth=3, label='4')
plt.plot(x, testingCurve[3], c='g', linewidth=2, label='8')
plt.plot(x, testingCurve[4], c='g', linestyle = '--', linewidth=3, label='12')
plt.plot(x, testingCurve[5], c='r', linewidth=2, label='16')
plt.plot(x, testingCurve[6], c='r', linewidth=2, label='32')
plt.title("testing curve using LDA")
plt.xlabel("percentage %")
plt.ylabel("accuracy")   
plt.ylim((0,1.05))
plt.legend(loc=(4))
plt.show()    