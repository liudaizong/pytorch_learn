import torch.nn as nn
import torch.nn.functional as f 
import torch
from torch.autograd import Variable
import torch.optim as optim

class NET(nn.Module):
	"""docstring for ClassName"""
	def __init__(self):
		super(NET, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.f1 = nn.Linear(16*5*5, 120)
		self.f2 = nn.Linear(120, 84)
		self.f3 = nn.Linear(84, 10)

	def forward(self, x):
		x = f.max_pool2d(f.relu(self.conv1(x)),(2,2))
		x = f.max_pool2d(f.relu(self.conv2(x)),2)
		x = x.view(x.size(0),-1)
		x = f.relu(self.f1(x))
		x = f.relu(self.f2(x))
		x = self.f3(x)

		return x

net = NET()
params = list(net.state_dict())
print len(params)
for name, parameters in net.named_parameters():
	print(name, ':', parameters.size())

input_x = Variable(torch.rand(1, 1, 32, 32))
target_y = Variable(torch.arange(0,10))
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
out = net(input_x)
loss = criterion(out, target_y)
print out.size()
print loss
optimizer.zero_grad()
loss.backward()
optimizer.step()