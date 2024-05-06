import numpy
from devil_ import *

out_,div_z,div_q=[[6,'devil_block_creater',"weight_devil/devil_block_creater.pt"],[16,'shuffle_block',"weight_devil/shuffle_block.pt"],
[16,'matrix_block',"weight_devil/matrix_block.pt"],[100,'linear_space',"weight_devil/linear_space.pt"],[5,'devil_operation_sec',"weight_devil/devil_operation_sec.pt"],
[5,'devil_chiose',"weight_devil/devil_chiose.pt"]],devil_sub.devil_lod(),{}

for i,z in zip(out_,range(len(div_z))):
	model= NeuralNetwork(i[0]).to(device)
	model.model_loder(model,i[2])
	div_q[i[1]]=model 

class main_devil:
	def __new__(self,z,w,i=0,devil=devil_out_cal(div_q['devil_block_creater'],div_z[0],'devil_block_creater')[0]):
		if w=='E':
			if devil[i]==1:
				z=main_devil.linear_devil(self,z,"E")
			elif devil[i]==2:
				z=main_devil.swap_devil(self,'E',z)
			elif devil[i]==3:
				z=operation_devil(z,devil_out_cal(div_q['devil_operation_sec'],div_z[4],'devil_operation_sec')[0])
		if w=='D':
			devil=devil[::-1]
			if devil[i]==1:
				z=main_devil.linear_devil(self,z,"D")
			elif devil[i]==2:
				z=main_devil.swap_devil(self,'D',z)
			elif devil[i]==3:
				z=operation_devil(z,devil_out_cal(div_q['devil_operation_sec'],div_z[4],'devil_operation_sec')[0][::-1],'D')
		if i<len(devil)-1:
			return main_devil(z,w,i+1,devil)
		else:
			return z

	def linear_devil(self,z,w):
		devil=devil_out_cal(div_q['linear_space'],div_z[3],'linear_space')[0]
		if "E"==w:
			for i in range(len(z)):
				for j in range(len(z[i])):
					z[i][j]=devil[z[i][j]]
		if "D"==w:
			for i in range(len(z)):
				for j in range(len(z[i])):
					for y in range(len(devil)):
						if z[i][j]==devil[y]:
							z[i][j]=y	
							break	
		return z

	def swap_devil(self,w,z):
		q=devil_out_cal(div_q['shuffle_block'],div_z[1],'shuffle_block')[0]
		f_devil,z=numpy.array([0]*16),z.reshape(-1)
		if w=='E':
			for i,j in zip(z,q):
				f_devil[j]=i
			return f_devil.reshape(4,4)
		if w=='D':
			for i,j in zip(range(len(z)),q):
				f_devil[i]=z[j]
			return f_devil.reshape(4,4)

class operation_devil:
	def __new__(self,z,order,q='E'):
		for i in order:
			y=devil_out_cal(div_q['matrix_block'],div_z[2],'matrix_block')[0].astype('int')
			if "E"==q:
				if '+'==i:
					z=operation_devil.add_devil(self,z,y)
				elif '-'==i:
					z=operation_devil.sub_devil(self,z,y)
				elif 'x'==i:
					z=operation_devil.mul_devil(self,z,y)
				elif 'XOR'==i:
					z=operation_devil.xor_devil(self,z,y)

			elif "D"==q:
				if '+'==i:
					z=operation_devil.sub_devil(self,z,y).astype('int')
				elif '-'==i:
					z=operation_devil.add_devil(self,z,y).astype('int')
				elif 'x'==i:
					z=operation_devil.rev_mul_devil(self,z,y).astype('int')
				elif 'XOR'==i:
					z=operation_devil.xor_devil(self,z,y)	.astype('int')
		return z

	def add_devil(self,z,y):
		return z+y
	def sub_devil(self,z,y):
		return z-y
	def mul_devil(self,z,y):
		return numpy.matmul(z,y)
	def xor_devil(self,z,y):
		return numpy.bitwise_xor(z,y)
	def rev_mul_devil(self,z,y):
		return numpy.ndarray.round(numpy.matmul(numpy.array(z),numpy.linalg.inv(y)))

devil_block_z=numpy.array([[1,2,33,4],[37,45,7,56],[10,11,77,13],[97,15,16,99]])
z=main_devil(devil_block_z,'E')
print(f"chiper matrix: {z}")
print(f"plain matrix: {main_devil(z,'D')}")