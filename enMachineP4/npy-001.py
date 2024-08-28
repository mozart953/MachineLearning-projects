import numpy as np

arreglo = np.array([1,2,3,4])
arreglo1 = np.arange(1,7)
print(arreglo)

print(arreglo1)
print(arreglo1[3:])

b = arreglo[1:]

print(b)

b[0]=40

print(arreglo)

r = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(r)
