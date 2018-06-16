'''
Plotting script for CIFAR-10/100
Copyright Xiaocheng Yang, 2018
'''


import numpy as np
import matplotlib.pyplot as plt

NN = "vgg19_bn"
DATA = "cifar10"
PATH_str = "checkpoints/cifar10/vgg19_bn/log-temp{}.txt"
TEMPS = [0.125, 0.25, 0.5, 1.0, 4.0, 16.0]
TRAINSIZE = 3

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']


def main():
	plt.figure(figsize=(18, 8))
	plt.title('{}-{} with different temps'.format(
        DATA, NN))
	plt.xlabel("Epochs")
	plt.ylabel("Error (%)")
	for i, temp in enumerate(TEMPS):
		with open(PATH_str.format(temp)) as f:
			lines = (line.rstrip() for line in f)
			Data = np.loadtxt(lines, delimiter='\t', skiprows=1).transpose()
			train_error = np.subtract(100, Data[3])
			vali_error = np.subtract(100, Data[4])

			plt.plot(range(TRAINSIZE), train_error, ':', color=COLORS[i],
	             label="Training error with temp {}".format(temp))

			plt.plot(range(TRAINSIZE), vali_error, '-', color=COLORS[i],
	             label="vali_error error with temp {}".format(temp))

			print(train_error)

	filename = '{}-{}.png'.format(
        DATA, NN)
	plt.legend(loc="best")
	plt.savefig(filename)


if __name__ == '__main__':
    main()
