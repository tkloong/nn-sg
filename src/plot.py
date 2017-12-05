import re
import matplotlib.pyplot as plt 
from matplotlib.pyplot import *
import argparse

# Parse argument
parser = argparse.ArgumentParser()
parser.add_argument('-log', required=True, type=str)
parser.add_argument('-s', '--saveFileName', required=True,  type=str)
args = parser.parse_args()

filename = [args.log]

#saveFileName = "sg_mmt_1e-2_0.9_train_test_loss.png"
saveFileName = args.saveFileName
title = re.search("(.*).png$", saveFileName).group(1)

color = ['r-', 'b-']
fig = [0, 0]

trainLoss = []
testLoss = []
y = []

# Parse 'newton'
with open(filename[0]) as fp:
    data = fp.readlines()
    for line in data:
        n = re.search(".*'train_loss': (\d*.\d*),", line)
        if n:
            trainLoss.append(float(n.group(1)))
            print(n.group(1))
        m = re.search(".*'test_loss': (\d*.\d*),", line)
        if m:
            testLoss.append(float(m.group(1)))
            print(m.group(1))

    y = [s*100 for s in range(0,len(trainLoss))]
    ytest = [s*100 for s in range(0,len(testLoss))]

    fig[0], = plt.plot(y, trainLoss, color[0])
    fig[1], = plt.plot(ytest, testLoss, color[1])

    #plt.axis([0, 400000, 0, 1.5])
    plt.ylabel('Obj. Value')
    plt.xlabel('Iteration')
    plt.title(title)
    #plt.show()
    #fig.savefig(saveFileName[i])

plt.legend([fig[0], fig[1]], ['train loss', 'test loss'], loc=1)
#plt.show()
savefig(saveFileName)
