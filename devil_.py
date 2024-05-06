import os
import numpy
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = ("cpu")
print(f"Using {device} device")

class devil_out_cal:
	def __new__(self,model,devil_t,model_n):
		devil,z=model(torch.from_numpy(numpy.array([devil_t]).astype('float32'))).detach(),[['devil_block_creater',devil_out_cal.devil_block_creater],
		['shuffle_block',devil_out_cal.shuffle_block],['matrix_block',devil_out_cal.matrix_block],['linear_space',devil_out_cal.linear_space],
		['devil_operation_sec',devil_out_cal.devil_operation_sec],['devil_chiose',devil_out_cal.devil_chiose]]
		return [i[1](self,devil) for i in z if model_n==i[0]][0],devil_sub.devil_in(devil,devil_t)

	def devil_block_creater(self,devil):
		return [1]+devil_sub.devil_high_ras([i for i in zip([2,2,2,3,3,3],numpy.array(torch.sin(devil)))])
	def shuffle_block(self,devil):
		return devil_sub.devil_high_ras([i for i in zip(range(0,16),numpy.array(torch.sin(devil)))])
	def matrix_block(self,devil):
		return numpy.array(devil).reshape(4,4)
	def linear_space(self,devil):
		return numpy.array(devil_sub.devil_high_ras([i for i in zip(range(0,100),numpy.array(torch.sin(devil)))]))#.reshape(10,10)
	def devil_operation_sec(self,devil):
		return numpy.array(devil_sub.devil_high_ras([i for i in zip(['+','-','x','/','XOR'],numpy.array(torch.sin(devil)))]))
	def devil_chiose(self,devil):
		return devil_sub.devil_high_ras([i for i in zip(range(1,6),numpy.array(torch.sin(devil)))])[0]
	
class devil_sub:
	def devil_high_ras(devil):
	  	left,right,demon_=devil[0][1],devil[0][1],[devil[0]]
	  	for i in devil:
	  		if i[1]<left:
	  			demon_.insert(0,i)
	  			left=i[1]
	  		elif i[1]>right:
	  			demon_.append(i)
	  			right=i[1]
	  		else:
	  			for j in range(0,len(demon_)):
	  				if i[1]<demon_[j][1]:
	  					demon_.insert(j,i)
	  					break
	  	return [i[0] for i in demon_]

	def devil_in(devil_z,devil_t):
		z=numpy.sum(numpy.array(devil_z))
		if z<=10 and z>=3:
			devil_t+=int(z)
		elif z<2:
			devil_t+=5
		else:
			devil_t+=int(str(int(z))[0:2])
		return devil_t
	def devil_sav(list_):
		open('catch_devil/devil_catch.txt',"wb").write('\n'.join([str(i) for i in list_]).encode('utf-8'))
	def devil_lod():
		return [int(i) for i in open('catch_devil/devil_catch.txt',"rb").read().decode('utf-8').split('\n')]

class NeuralNetwork(nn.Module):
    def __init__(self,oup_):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, oup_),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

    def train_devil(self,model,out_):
    	x_train=torch.from_numpy(numpy.random.rand(100,1).astype('float32'))
    	y_train=torch.from_numpy(numpy.random.rand(100,out_).astype('float32'))
    	criterion = nn.MSELoss()
    	optimizer = optim.Adam(model.parameters(), lr=0.001)
    	for epoch in range(1000):
    		outputs = model(x_train)
    		loss = criterion(outputs,y_train)
    		loss.backward()
    		optimizer.step()
    		print(f'[{epoch}, loss: {loss}')
    def model_saver(self,model,devil_fn):
    	torch.save(model.linear_relu_stack.state_dict(),devil_fn)
    def model_loder(self,model,devil_fn):
    	model.linear_relu_stack.load_state_dict(torch.load(devil_fn))

if __name__=="__main__":
	
	out_,div_z=[[6,'devil_block_creater',"weight_devil/devil_block_creater.pt"]],devil_sub.devil_lod()
	print(div_z)

	for i,z in zip(out_,range(len(div_z))):
		model= NeuralNetwork(i[0]).to(device)
		model.train_devil(model,i[0])
		model.model_saver(model,i[2])
		#model.model_loder(model,i[2])
		angel=devil_out_cal(model,div_z[z],i[1])
		div_z[z]=angel[1]
		print(angel[0])
	devil_sub.devil_sav(div_z)