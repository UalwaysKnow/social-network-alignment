import numpy as np
import random

s=[]
f = open('twit.txt','r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split('\t')
    #字典里values是列表
    s.append(int(line[0]))
    s.append(int(line[1]))
f.close()

a=np.zeros(shape=(3,14808))
b=np.zeros(shape=(1,14808))
for j in range(14808):
    a[0][j]=j
for i in range(14808):
    for j in range(len(s)):
        if s[j]==i:
            b[0][i]=b[0][i]+1
a[1]=b[0]
count=0
for m in range(14808):
    count=b[0][m]+count
for n in range(14808):
    a[2][n]=b[0][n]/count
np.savetxt('twit-gailv.txt',a)
print(a)
