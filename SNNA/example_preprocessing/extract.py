import numpy as np
s=[]
f = open('anchors_test.txt','r')
lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split('\t')
    #字典里values是列表
    s.append(line[0])
f.close()
a = np.loadtxt('facebook-paixu.txt')
b=np.zeros(shape=(1053,101))
for i in range(1053):
    x=int(s[i])
    b[i]=a[x]
b=b.astype(np.float32)
np.savetxt('facebook-align-test.txt',b)
c=np.shape(b)
print(b)
print(c)
