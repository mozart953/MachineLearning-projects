import random

entrada1 = random.random()
entrada2 = random.random()
entrada3 = random.random()

peso1 = 0.91823
peso2 = 0.190237
peso3 = 1.29387
umbral = 0.123897

salida = peso1*entrada1 + peso2*entrada2 + peso3*entrada3
salida += umbral
print(salida)

print("Hello World")