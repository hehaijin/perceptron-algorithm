import csv
import re
import copy
import numpy as np
import time

#the function
def calculate(W,P):
	length=len(P)
	result=0
	for i in range(length):
		if i==0:
			result=result+W[i]*1
		else:
			result=result+W[i]*P[i]
	if result >0:
		result=1
	else:
		result=0
	return result

#W, parameter vector, P, data point
def updateParameter(W,P):
	length=len(P) 
	label=P[0]
	result=calculate(W,P)
	diff=label-result
	for i in range(length):
		if i==0:
			W[i]=W[i]+diff*1
		else:
			W[i]=W[i]+diff*P[i]	
	return W
	
#label is 0 or 1	
def trainingBinary(data):
	W=[]
	length=len(data[0])
	for i in range(length):
		W.append(0)
	for i in range(len(data)):
		W=updateParameter(W,data[i])
	diffsum=validateBinary(W,data)
	repeat=0
	#print("first round diff "+ str(diffsum))
	while diffsum >20 and repeat <0:
		for i in range(len(data)):
			W=updateParameter(W,data[i])
		diffsum=validateBinary(W,data)
		repeat=repeat+1
		print(diffsum)
	return W
	

def validateBinary(W,data):
	diffsum=0
	length=len(data[0])
	for i in range(len(data)):
		output=calculate(W,data[i])
		diff=data[i][0]-output
		diffsum=diffsum+abs(diff)	
	return diffsum
	
	
	
		

#deal with 4 class labels
def trainingMultiple(data1):
	
	length=len(data1[0])
	result=[]
	#divide data
	for i in range(4):
		data=copy.deepcopy(data1)
		#print("deep copy")
		for j in range(len(data)):
			if data[j][0]!=i+1:
				data[j][0]=0
			else:
				data[j][0]=1
		#print("training start")		
		w=trainingBinary(data)
		#print("training end")
		result.append(w)
	return result

	
def validateMultiple(W, data):
	count=0
	length=len(data[0])
	for i in range(len(data)):
		r=[]
		for j in range(4):
			result=0
			for t in range(length):
				if t==0:
					result=result+1*W[j][0]
				else:
					result=result+W[j][t]*data[i][t]
			r.append(result)
		#print(r)
		m=max(r)
		for j in range(4):
			if r[j]==m:
				plabel=j+1
		if plabel==data[i][0]:
			count=count+1
	return count/len(data)
	
	
	 



def readData():
	data=[]
	with open('data.txt', newline='') as csvfile:
			for line in csvfile:
				sp=re.split(" +",line)
				p=[]
				for i in range(len(sp)-1):
					p.append(float(sp[i+1]))
				data.append(p)
	return data
			
			
			
def classify(data):
					
	start_time=time.time()
	#10 fold cross validation
	accurancy=[]
	for i in range(10):
		
		training=[]
		testing=[]
		for j in range(len(data)):
			if j%10 ==i:
				testing.append(data[j])
			else: 
				training.append(data[j])
		W=trainingMultiple(training)
		
		c=validateMultiple(W,testing)
		#print("validation accurancy "+ str(c))
		accurancy.append(c)
	avg=0
	for i in range(10):
		avg=avg+accurancy[i]	
	avg=avg/10	
	print("the avarage accurancy of 10 fold cross validation is "+ str(avg)) 	
	print("time taken is "+ str(time.time()-start_time))

#use harr transform to recude data dimension
def harrReduced(data,n):
	data=np.array(data)
	N=[8,8,4,4,2,2,2,2]
	N=np.array(N)
	Haar=[[1,1,1,1,1,1,1,1],[1,1,1,1,-1,-1,-1,-1],[1,1,-1,-1,0,0,0,0],[0,0,0,0,1,1,-1,-1],[1,-1,0,0,0,0,0,0],[0,0,1,-1,0,0,0,0],[0,0,0,0,1,-1,0,0],[0,0,0,0,0,0,1,-1]]
	Haar=np.array(Haar)
	data1=np.dot(data[:,1:],Haar.T)
	#print(data1[0])
	data1=data1/N
	#print(data1[0])
	data2=data1[:,0:n]
	data3=np.concatenate((data[:,0:1],data2),axis=1)
	#print(data3)
	return data3




#starting the execution


data=readData()
#print(data)
print("original data")
classify(data)
for i in range(7,1,-1):
	data2=harrReduced(data,i)
	print("on reduced data to dimension "+ str(i))
	classify(data2)





