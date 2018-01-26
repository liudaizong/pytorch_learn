import torchvision as tv 
import torchvision.transforms as transforms 
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.autograd import Variable
import torch.utils.data as data

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

trainset = tv.datasets.CIFAR10(
	root = './CIFAR10',
	train = True,
	download = True,
	transform = transform)

trainloader = data.DataLoader(
	trainset,
	batch_size = 4,
	shuffle = True,
	num_workers = 2)

testset = tv.datasets.CIFAR10(
	root = './CIFAR10',
	train = False,
	download = False,
	transform = transform)

testloader = data.DataLoader(
	testset,
	batch_size = 4,
	shuffle = False,
	num_workers = 2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
