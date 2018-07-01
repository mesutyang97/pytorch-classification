python cifar.py -d cifar100 -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar100/vgg19_bn-0.1 --gpu-id 1 --temp 0.1
python cifar.py -d cifar100 -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar100/vgg19_bn-1 --gpu-id 1 --temp 1.0
python cifar.py -d cifar100 -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar100/vgg19_bn-3 --gpu-id 1 --temp 3.0
python cifar.py -d cifar100 -a vgg19_bn --epochs 164 --schedule 81 122 --gamma 0.1 --checkpoint checkpoints/cifar100/vgg19_bn-10 --gpu-id 1 --temp 10.0
