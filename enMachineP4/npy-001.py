import math

import numpy as np
import math as mt

arreglo = np.array([1,2,3,4])
arreglo1 = np.arange(1,7)
print(arreglo)

print(arreglo1)
print(arreglo1[3:])

b = arreglo[1:]

print(b)

b[0]=40

print(arreglo)


r = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12],[2, 8, 7, 3]])

print(r)

print(f"Valor en fila 2 y columna 1: {r[2,1]}")

print(f"Dimension del arreglo es: {r.ndim}")

print(arreglo.ndim)
print(r.shape)
print(r.size)

print(r.size == mt.prod(r.shape))

print(r.dtype)


#Bsic arrays

print(np.zeros(5))
print(np.ones(5))
print(np.empty(3))
print(np.linspace(0,10, num=5))

print(np.ones(2, dtype=np.int64))

array8 = np.array([8,7,6,5,4,3,2])
print(np.sort(array8))

array9 = np.arange(7,10,2)
array10 = np.arange(12,20.2)

array11 = np.concatenate((array9, array10))
print(array11)
