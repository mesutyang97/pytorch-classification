'''
Plotting script for CIFAR-10/100
Copyright Xiaocheng Yang, 2018
'''


import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

# Base_PATH = "../code/ts/cifar/checkpoints/"
Base_PATH = "checkpoints/{}/{}_{}/log.txt"
Graph_PATH = "graph"
TEMPS = [0.1, 0.3, 1, 3, 10, 30]

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/100 Result Plotting')
# Datasets
parser.add_argument('-d', '--dataset', default='cifar10', type=str)

# Architecture
parser.add_argument('--arch', '-a', default='vgg19_bn', type=str)

args = parser.parse_args()


def trainingCurve(dataset, arch):
	plt.figure(figsize=(18, 8))
	plt.title('{} {} training curve with different temps'.format(
        dataset, arch))
	plt.xlabel("Epochs")
	plt.ylabel("Error (%)")
	for i, temp in enumerate(TEMPS):
		with open(Base_PATH.format(dataset, arch, temp)) as f:
			lines = (line.rstrip() for line in f)
			Data = np.loadtxt(lines, delimiter='\t', skiprows=1).transpose()
			train_error = np.subtract(100, Data[3])
			vali_error = np.subtract(100, Data[4])
			trainsize = len(train_error)

			plt.plot(range(trainsize), train_error, ':', color=COLORS[i],
	             label="Training error with temp {}".format(temp))

			plt.plot(range(trainsize), vali_error, '-', color=COLORS[i],
	             label="vali_error error with temp {}".format(temp))

	filename = 'training_curve-{}-{}-{}.png'.format(
        dataset, arch, time.time())
	plt.legend(loc="best")
	plt.savefig(filename)



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
        dataset, arch, time.time())
	plt.legend(loc="best")
	plt.savefig(filename)



def trainingCurveZoomed(dataset, arch):
	plt.figure(figsize=(18, 8))
	plt.title('{} {} training curve with different temps, Zoomed'.format(
        dataset, arch))
	plt.xlabel("Epochs")
	plt.ylabel("Error (%)")
	for i, temp in enumerate(TEMPS):
		with open(Base_PATH.format(dataset, arch, temp)) as f:
			lines = (line.rstrip() for line in f)
			Data = np.loadtxt(lines, delimiter='\t', skiprows=1).transpose()
			train_error = np.subtract(100, Data[3])
			vali_error = np.subtract(100, Data[4])
			trainsize = len(train_error)

			plt.plot(range(trainsize), train_error, ':', color=COLORS[i],
	             label="Training error with temp {}".format(temp))

			plt.plot(range(trainsize), vali_error, '-', color=COLORS[i],
	             label="vali_error error with temp {}".format(temp))

	filename = 'training_curve_zoommed-{}-{}-{}.png'.format(
        dataset, arch, time.time())
	axes = plt.gca()
	axes.set_ylim([0,30])
	plt.legend(loc="best")
	plt.savefig(filename)

def main():
	trainsize = 164
	if args.arch == 'densenet-bc-100-12':
		trainsize = 300
	#errorCurve(args.dataset, args.arch, trainsize)
	#trainingCurve(args.dataset, args.arch)
	trainingCurveZoomed(args.dataset, args.arch)

if __name__ == '__main__':
    main()
