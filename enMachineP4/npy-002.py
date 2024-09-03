import pandas as pd

numeros = [1,2,3,4,5,6,7,8,9,10,2,3]

#pd.DataFrame()

lista = pd.Series(numeros)
print(lista)

print(lista.head(3))
print(lista.tail(2))
print(f"El valor maximo es: {lista.max()}")
print(f"El valor minimo es: {lista.min()}")
print(f"El valor promedio es: {lista.mean()}")

#obtener datos
print(lista.get(4))

#Filtrar un array
filtro=lista==2
print(lista[filtro])

#serie2
b1=[23,15,12,25,20,50,31,2]

serie2=pd.Series(b1)
print(serie2)

print(lista+serie2)

b1 = serie2.sort_index(ascending=False)
print(b1)

def cuadrado(x):
    return x*x

bb2=serie2.map(cuadrado)

print(bb2)