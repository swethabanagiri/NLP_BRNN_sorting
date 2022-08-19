import random as r
total=10000

f=open('inputs.txt','a')

for i in range(total):
	st=r.sample(xrange(0,32),r.randint(2,11))
	f.write(str(st))
	f.write("\n")