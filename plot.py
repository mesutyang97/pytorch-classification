'''
Plotting script for CIFAR-10/100
Copyright Xiaocheng Yang, 2018
'''


import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

Base_PATH = "../code/ts/cifar/checkpoints/"
# Base_PATH = "checkpoints/{}/{}_{}/log.txt"
Graph_PATH = "graph"
TEMPS = [0.1, 0.3, 1, 3, 10, 30]

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Result Plotting')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)

# Architecture
parser.add_argument('--arch', '-a', default='vgg19_bn', type=str)

args = parser.parse_args()

def errorCurve(dataset, arch, trainsize):
	plt.figure(figsize=(8, 8))
	plt.title('{} {} final error with different temps'.format(
        dataset, arch))
	plt.xlabel("log(temps)")
	plt.ylabel("Error (%)")
	train_error_temps = []
	vali_error_temps = []

	logtemp = np.log(TEMPS)

	for i, temp in enumerate(TEMPS):
		with open(Base_PATH.format(dataset, arch, temp)) as f:
			lines = (line.rstrip() for line in f)
			Data = np.loadtxt(lines, delimiter='\t', skiprows=1).transpose()
			train_error_temps.append(100 - Data[3][-1])
			vali_error_temps.append(100 - Data[4][-1])

	
	plt.plot(logtemp, train_error_temps, ':', color=COLORS[i],
	        label="Training error")

	plt.plot(logtemp, vali_error_temps, '-', color=COLORS[i],
	        label="vali_error error")

	axes = plt.gca()
	axes.set_ylim([0,30])
	filename = 'final_error-{}-{}-{}.png'.format(
        dataset,arch, time.time())
	plt.legend(loc="best")
	plt.savefig(filename)
def main():
	# trainingCurve()
	trainsize = 164
	if args.arch == 'densenet-bc-100-12':
		trainsize = 300
	errorCurve(args.dataset, args.arch, trainsize)

if __name__ == '__main__':
    main()
