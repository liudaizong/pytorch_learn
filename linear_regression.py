import torch as t 
from torch.autograd import Variable as v 
from matplotlib import pyplot as plt 

t.manual_seed(1000)

def get_fake_data(batch_size = 8):
	x = t.rand(batch_size, 1) * 20
	y = x * 2 + (1 + t.rand(batch_size, 1)) * 3
	return x, y

w = v(t.rand(1, 1), requires_grad = True)
b = v(t.zeros(1, 1), requires_grad = True)
lr = 0.001

for ii in range(1000):
	x, y = get_fake_data()
	x, y = v(x), v(y)

	y_pred = x.mm(w) + b.expand_as(y)
	loss = 0.5 * (y_pred - y) ** 2
	loss = loss.sum()

	loss.backward()

	w.data.sub_(lr * w.grad.data)
	b.data.sub_(lr * b.grad.data)
	w.grad.data.zero_()
	b.grad.data.zero_()

print w.data.squeeze()[0], b.data.squeeze()[0]
x, y = get_fake_data()
plt.scatter(x.squeeze().numpy(), y.squeeze().numpy())
x = t.arange(0, 20).view(-1, 1)
y = x.mm(w.data) + b.data.expand_as(x)
plt.plot(x.squeeze().numpy(), y.squeeze().numpy())
plt.show()

