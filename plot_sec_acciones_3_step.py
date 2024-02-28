#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:10:29 2024

@author: sabal
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Supongamos que tienes un archivo pickle llamado 'mi_diccionario.pkl'
archivo_path = "/home/sabal/N_STEP_DQN_COMPANION-main/Checkpoints/fold5_DQN_update_rate_100_metiendo_borde_3_STEPS_02-02-2024_19:40_step_by_step_SEED_97_MEM_SIZE_20000NO_ACTION_PROB_0_BETA_1.25_LAMBDA_0.1"
archivo_pickle =archivo_path+'/dict_actions_G.pkl'

# Cargar el diccionario desde el archivo pickle
with open(archivo_pickle, 'rb') as archivo:
    mi_diccionario_cargado = pickle.load(archivo)

# Imprimir el diccionario cargado
conteo_por_clave = {}

#Recorrer las claves y contar la cantidad de elementos en cada lista
for clave, valor in mi_diccionario_cargado.items():
    if isinstance(valor, list):  # Verificar si el valor es una lista
        cantidad_elementos = len(valor)
        if cantidad_elementos > 900:
            conteo_por_clave[str(clave)] = cantidad_elementos
    # else:
    #     conteo_por_clave[clave] = 0  # Otra opción, si el valor no es una lista, asignar 0

# Imprimir el conteo de elementos de mayor a menor
conteo_ordenado = sorted(conteo_por_clave.items(), key=lambda x: x[1], reverse=True)

# Ordenar las claves y los valores por el valor de las barras
claves_ordenadas, valores_ordenados = zip(*sorted(conteo_por_clave.items(), key=lambda x: x[1], reverse=True))


# Ajustar el tamaño de la figura
fig = plt.figure(figsize=(12, 6))

# Crear un gráfico de barras con etiquetas rotadas
barras = plt.bar(claves_ordenadas, valores_ordenados, color='blue')
plt.xlabel('Acciones')
plt.ylabel('Cantidad de veces que se ha hecho la secuencia')
plt.title('Conteo de veces que se ha hecho esa secuencia de acciones en 400 epocas')
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x y ajustar la alineación
# Mostrar el valor numérico de cada barra
for barra, valor in zip(barras, valores_ordenados):
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.1, str(valor), ha='center')

plt.tight_layout()  # Ajustar el diseño para evitar recortes

plt.show()
fig.savefig(archivo_path+'/veces_sec_acciones.png',dpi=300)


# fig.savefig('/home/sabal/N_STEP_DQN_COMPANION-main/Checkpoints/PICKLE_SEC_ACCIONES_fold1_DQN_update_rate_1003_STEPS_29-01-2024_15:22_step_by_step_SEED_97_MEM_SIZE_20000NO_ACTION_PROB_0_BETA_1.25'+'/veces_sec_acciones.png',dpi=300)




for clave, cantidad_elementos in conteo_ordenado:
    print(f"Clave: {clave}, Cantidad de Elementos: {cantidad_elementos}")
    
conteo_por_clave = {}
valor_a_contar = 0
# Recorrer las claves y contar la cantidad de veces que aparece el valor en cada lista
for clave, lista_elementos in mi_diccionario_cargado.items():
    if isinstance(lista_elementos, list):
        cantidad = lista_elementos.count(valor_a_contar)
        if cantidad != 0:
            conteo_por_clave[str(clave)] = cantidad

conteo_ordenado = sorted(conteo_por_clave.items(), key=lambda x: x[1], reverse=True)

# Imprimir el resultado
print(f"El valor {valor_a_contar} aparece en las listas asociadas a las claves de la siguiente manera:")
for clave, cantidad in conteo_ordenado:
    if cantidad != 0:
        print(f"Clave: {clave}, Cantidad: {cantidad}")
        
# Crear un gráfico de barras
claves = list(conteo_por_clave.keys())
valores = list(conteo_por_clave.values())



# Ordenar las claves y los valores por el valor de las barras
claves_ordenadas, valores_ordenados = zip(*sorted(conteo_por_clave.items(), key=lambda x: x[1], reverse=True))


# Ajustar el tamaño de la figura
fig = plt.figure(figsize=(12, 6))

# Crear un gráfico de barras con etiquetas rotadas
barras = plt.bar(claves_ordenadas, valores_ordenados, color='blue')
plt.xlabel('Acciones')
plt.ylabel('Cantidad de veces que aparece el valor')
plt.title(f'Conteo del valor G = {valor_a_contar} para cada secuencia de acciones')
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x y ajustar la alineación
# Mostrar el valor numérico de cada barra
for barra, valor in zip(barras, valores_ordenados):
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.1, str(valor), ha='center')

plt.tight_layout()  # Ajustar el diseño para evitar recortes

plt.show()
fig.savefig(archivo_path+"/veces_G_0.png",dpi=300)



# Crear un diccionario para almacenar el conteo por clave
conteo_por_clave = {}

# Recorrer las claves y contar la cantidad de veces que aparece el valor en cada lista
for clave, lista_elementos in mi_diccionario_cargado.items():
    if isinstance(lista_elementos, list):
        cantidad = lista_elementos.count(valor_a_contar)
        if cantidad != 0:
            conteo_por_clave[str(clave)] = {"total": len(lista_elementos), "cumple_condicion": cantidad}

# Ordenar las claves según el total
claves_ordenadas = sorted(conteo_por_clave.keys(), key=lambda x: conteo_por_clave[x]["total"], reverse=True)

# Obtener los valores ordenados
valores_total = [conteo_por_clave[clave]["total"] for clave in claves_ordenadas]
valores_cumplen_condicion = [conteo_por_clave[clave]["cumple_condicion"] for clave in claves_ordenadas]

# Ajustar el tamaño de la figura
fig = plt.figure(figsize=(12, 6))

# Crear un gráfico de barras apiladas
barra_total = plt.bar(claves_ordenadas, valores_total, color='red', label='Total')
barra_condicion = plt.bar(claves_ordenadas, valores_cumplen_condicion, color='green', label='G = 0')

plt.xlabel('Secuencia de cciones')
plt.ylabel('Cantidad de veces que aparece el valor')
plt.title('Conteo de veces que se ha hecho esa secuencia de acciones en 400 epocas')
plt.xticks(rotation=45, ha='right')  # Rotar las etiquetas del eje x y ajustar la alineación
for barra, valor in zip(barra_total, valores_total):
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.1, str(valor), ha='center')
for barra, valor in zip(barra_condicion, valores_cumplen_condicion):
    plt.text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.1, str(valor), ha='center')

plt.legend(loc='upper right')  # Agregar la leyenda
plt.tight_layout()  # Ajustar el diseño para evitar recortes
plt.show()

fig.savefig(archivo_path+"/veces_G_0_vs_total.png",dpi=300)

# fig.savefig('/home/sabal/N_STEP_DQN_COMPANION-main/Checkpoints/PICKLE_SEC_ACCIONES_fold1_DQN_update_rate_1003_STEPS_29-01-2024_15:22_step_by_step_SEED_97_MEM_SIZE_20000NO_ACTION_PROB_0_BETA_1.25'+'/veces_G_0_vs_total.png',dpi=300)
