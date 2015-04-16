
from array import array
a = [0.0] * 990 * 100000
f = open('990_in.dat','wb')
fa = array('d',a)
fa.tofile(f)
f.close()
