import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import KNN

datingDataMat, datingLables = KNN.file2matrix('datingTestSet2.txt')
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLables), 15.0 * array(datingLables))
plt.show()
