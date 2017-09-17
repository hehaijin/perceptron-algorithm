import csv
import re



with open('data.txt', newline='') as csvfile:
		for line in csvfile:
			sp=re.split(" +",line)
			data=[]
			for i in range(len(sp)-1):
				data.append(float(sp[i+1]))
			print(data)	
