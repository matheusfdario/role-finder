
import numpy as np

def myfunc(x, y, z):
    return [x, y, z]

dim0 = 3
dim1 = dim0
dim2 = dim1

r = np.fromfunction(myfunc, (dim0,dim1,dim2))

r_0 = r[0]
r_1 = r[1]
r_2 = r[2]

for i in range(dim0):
    for j in range(dim1):
        for k in range(dim2):
            print('('+r_0[i,j,k].__str__()+','+r_1[i,j,k].__str__()+','+r_2[i,j,k].__str__()+')')
