#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 09:29:12 2023

@author: sabal
"""
import math
import os
import glob
import gym
from itertools import count
import random
import matplotlib

import matplotlib.pyplot as plt
import numpy as np

import pdb 
from datetime import datetime
import pandas as pd
import pickle

# file = '/results_test_raw_'+NUM_LOOP+'.csv'

def proccessCsv(path, NUM_LOOP,mode):
    if mode == 'test':
        file = '/results_test_raw_'+NUM_LOOP+'.csv'
    else:
        file = '/results_val_raw.csv'
    data = pd.read_csv(path+file)
    
    data['energy_reward'] = data['energy_reward'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['energy_reward_episode'] = data['energy_reward'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    data['energy_reward_epoch'] = data['energy_reward_episode'].apply(lambda x: np.mean(x))
    
    data['time_reward'] = data['time_reward'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['time_reward_episode'] = data['time_reward'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    data['time_reward_epoch'] = data['time_reward_episode'].apply(lambda x: np.mean(x))
    
    
    data['reward'] = data['reward'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['reward_episode'] = data['reward'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    data['reward_epoch'] = data['reward_episode'].apply(lambda x: np.mean(x))
    
    # data['G'] = data['G'].apply(eval)
    # # Calcular la media de cada sublista en la columna 'energy_reward'
    # data['G_episode'] = data['G'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    # data['G_epoch'] = data['G_episode'].apply(lambda x: np.mean(x))
        
        
    data['delay'] = data['delay'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['delay_episode'] = data['delay'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    data['delay_epoch'] = data['delay_episode'].apply(lambda x: np.mean(x))
    
    
    data['minimum_delay'] = data['minimum_delay'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['minimum_delay_episode'] = data['minimum_delay'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    data['minimum_delay_epoch'] = data['minimum_delay_episode'].apply(lambda x: np.mean(x))
    
    
    data['reactive_delay'] = data['reactive_delay'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['reactive_delay_episode'] = data['reactive_delay'].apply(lambda x: list(map(lambda sublista: np.sum(sublista), x)))
    data['reactive_delay_epoch'] = data['reactive_delay_episode'].apply(lambda x: np.mean(x))
    
    
    data['reactive_time'] = data['reactive_time'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['reactive_time_episode'] = data['reactive_time'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['reactive_time_epoch'] = data['reactive_time_episode'].apply(lambda x: np.sum(x))
    
    
    data['minimum_time'] = data['minimum_time'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['minimum_time_episode'] = data['minimum_time'].apply(lambda x: list(map(lambda sublista: np.mean(sublista), x)))
    data['minimum_time_epoch'] = data['minimum_time_episode'].apply(lambda x: np.sum(x))
    
    data['interaction_time'] = data['interaction_time'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['interaction_time_episode'] = data['interaction_time'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['interaction_time_epoch'] = data['interaction_time_episode'].apply(lambda x: np.sum(x))
    
    data['CA_intime'] = data['CA_intime'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['CA_intime_episode'] = data['CA_intime'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['CA_intime_epoch'] = data['CA_intime_episode'].apply(lambda x: np.sum(x))
    
    data['CA_late'] = data['CA_late'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['CA_late_episode'] = data['CA_late'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['CA_late_epoch'] = data['CA_late_episode'].apply(lambda x: np.sum(x))
    
    data['IA_intime'] = data['IA_intime'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['IA_intime_episode'] = data['IA_intime'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['IA_intime_epoch'] = data['IA_intime_episode'].apply(lambda x: np.sum(x))
    
    data['IA_late'] = data['IA_late'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['IA_late_episode'] = data['IA_late'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['IA_late_epoch'] = data['IA_late_episode'].apply(lambda x: np.sum(x))
    
    data['CI'] = data['CI'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['CI_episode'] = data['CI'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['CI_epoch'] = data['CI_episode'].apply(lambda x: np.sum(x))
    
    data['II'] = data['II'].apply(eval)
    # Calcular la media de cada sublista en la columna 'energy_reward'
    data['II_episode'] = data['II'].apply(lambda x: list(map(lambda sublista: sublista[-1], x)))
    data['II_epoch'] = data['II_episode'].apply(lambda x: np.sum(x))
    
    
    data_save = data[['energy_reward_epoch','time_reward_epoch','reward_epoch','delay_epoch','minimum_delay_epoch','reactive_delay_epoch','reactive_time_epoch','minimum_time_epoch','interaction_time_epoch', 'CA_intime_epoch','CA_late_epoch','IA_intime_epoch','IA_late_epoch','CI_epoch','II_epoch']]
    if mode == 'test':
        data_save.to_csv(path+'/results_test_proccessed_'+NUM_LOOP+'.csv')
    else:
        data_save.to_csv(path+'/results_test_proccessed_val'+NUM_LOOP+'.csv')
    return data_save,data
                    
    
def plotActions (path, file):
     
     data = pd.read_csv(path+file)

     fig1 = plt.figure(figsize=(15, 8))

     plt.subplot2grid((2,3), (0,0))
     plt.title("Correct actions (in time)")
     plt.plot(list(data['CA_intime_epoch']), 'springgreen')

     plt.subplot2grid((2,3), (1,0))
     plt.title("Correct actions (late)")
     plt.plot(list(data['CA_late_epoch']), 'royalblue')
     plt.xlabel("Epoch")

     plt.subplot2grid((2,3), (0,1))
     plt.title("Incorrect actions (in time)")
     plt.plot(list(data['IA_intime_epoch']), 'red')

     plt.subplot2grid((2,3), (1,1))
     plt.title("Incorrect actions (late)")
     plt.plot(list(data['IA_late_epoch']), 'red')
     plt.xlabel("Epoch")

     plt.subplot2grid((2,3), (0,2))
     plt.title("Correct inactions")
     plt.plot(list(data['CI_epoch']), 'springgreen')

     plt.subplot2grid((2,3), (1,2))
     plt.title("Incorrect inactions")
     plt.plot(list(data['II_epoch']), 'red')
     plt.xlabel("Epoch")

     if 'val' in file:
         save_name ='/ACTIONS_val.jpg'
     else:
         save_name = '/ACTIONS_test.jpg'
     fig1.savefig(path+save_name)
     # plt.show()
     plt.close(fig1)
     
     
def visualizeSecActions (archivo_path):
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

    # plt.show()
    fig.savefig(archivo_path+'/veces_sec_acciones.png',dpi=300)

    # for clave, cantidad_elementos in conteo_ordenado:
    #     print(f"Clave: {clave}, Cantidad de Elementos: {cantidad_elementos}")
        
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
    # print(f"El valor {valor_a_contar} aparece en las listas asociadas a las claves de la siguiente manera:")
    # for clave, cantidad in conteo_ordenado:
    #     if cantidad != 0:
            # print(f"Clave: {clave}, Cantidad: {cantidad}")

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

    # plt.show()
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
    # plt.show()

    fig.savefig(archivo_path+"/veces_G_0_vs_total.png",dpi=300)
     
if( __name__ == '__main__' ):
    NUM_LOOP =  '0'
    EXP_NAME = 'fold5_DQN_update_rate_100_metiendo_borde_1_STEPS_02-02-2024_22:45_step_by_step_SEED_97_MEM_SIZE_20000NO_ACTION_PROB_0_BETA_1.25_LAMBDA_0.1'
    path = '/home/sabal/N_STEP_DQN_COMPANION-main/Checkpoints/'+EXP_NAME 
    data_save1,data1 = proccessCsv(path, NUM_LOOP, 'test')