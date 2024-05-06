import os
import numpy
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = ("cpu")
print(f"Using {device} device")
class devil_sub:
	def devil_in(devil_z,devil_t):
		z=numpy.sum(numpy.array(devil_z))
		if z<=10 and z>=3:
			devil_t+=int(z)
		elif z<2:
			devil_t+=5
		else:
			devil_t+=int(str(int(z))[0:2])
		return devil_t


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def train_devil(self,model):
    	x_train=torch.from_numpy(numpy.random.rand(100,1).astype('float32'))
    	y_train=torch.from_numpy(numpy.random.rand(100,16).astype('float32'))
    	criterion = nn.MSELoss()
    	optimizer = optim.Adam(model.parameters(), lr=0.001)
    	for epoch in range(1000):
    		outputs = model(x_train)
    		loss = criterion(outputs,y_train)
    		loss.backward()
    		optimizer.step()
    		print(f'[{epoch}, loss: {loss}')
    def model_saver(self,model):
    	torch.save(model.linear_relu_stack.state_dict(), "weight_devil/devil.pt")
    def model_loder(self,model):
    	model.linear_relu_stack.load_state_dict(torch.load("weight_devil/devil.pt"))



model = NeuralNetwork().to(device)
model.train_devil(model)
devil_t=int(input(":"))
for i in range(10):
	devil=model(torch.from_numpy(numpy.array([devil_t]).astype('float32'))).detach()
	print(numpy.array(devil).reshape(4,4))
	devil_t=devil_sub.devil_in(devil,devil_t)
	
