import numpy as np
a = np.loadtxt('four.embeddings')
b=np.zeros(shape=(5313,101))
for i in range(5313):
	for j in range(5313):
		if a[j][0]==i:
			b[i]=a[j]
			break
d = b.astype(np.float32)
np.savetxt('facebook-sorted.txt',d)
c=np.shape(b)
print(b)
print(c)
print(d)
