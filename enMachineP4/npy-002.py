import pandas as pd
import numpy as np

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

df2 = pd.DataFrame(
        {
            "A": 1.0,
            "B": pd.Timestamp("20130102"),
            "C": pd.Series(1, index=list(range(4)), dtype="float32"),
            "D": np.array([3] * 4, dtype="int32"),
            "E": pd.Categorical(["test", "train", "test", "train"]),
            "F": "foo",
        }
    )
print(df2)


datos = {
    "nombre":["Argentina", "Brazil", "Chile", "Colombia","Ecuador" ,"Paraguay", "Peru"],
    "capital":["Buenos Aires", "Brasilia", "Santiago de Chile", "Bogota","Quito", "Asuncion", "Lima"],
    "poblacion":[45380000, 212600000, 19120000,50880000, 17640000, 7133000, 32970000],
    "sup. en k2":[2780000, 8516000, 756950, 1143000, 283560, 406752, 1285216]
}

paises = pd.DataFrame(datos)

print(paises)

print("Primeros 4")
print(paises.head())
print("Ultimos 3")
print(paises.tail(3))

paises2 = pd.DataFrame(datos, columns=["nombre", "poblacion"])

print(paises2)

paises3 = pd.DataFrame(datos, index=datos["nombre"], columns=["capital", "poblacion", "sup. en k2"])
print(paises3)