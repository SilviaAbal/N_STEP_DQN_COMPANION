import torch
import torch.nn as nn
import random
from collections import namedtuple, deque
import config as cfg
import torch.nn.functional as F
import pdb 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import collections as matcoll

Transition = namedtuple('Transition',
                        ('pHidden', 'state', 'action', 'tHidden', 'next_state', 'reward'))


class ReplayMemory(object):

    """
    This class implements a first in first out buffer with maximum
    capacity "capacity". It is possible to sample 'batch_size' 
    elements (unique elements) from the queue
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        return
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        return
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def get_elements(self, x):
        # pdb.set_trace()
        for transition in list(self.memory)[:x]:
            print('***** State *****:\n ',transition.state)
            print('***** Action *****:\n ', transition.action)
            print('***** Reward *****:\n ', transition.reward)
            
            # pdb.set_trace()
        # print([('state: {},\n action: {}, \n  reward: {} \n '.format(transition.state, transition.action, transition.reward)) for transition in self.memory[:x]])
        # return [('state: {},\n action: {}, \n  reward: {} \n '.format(transition.state, transition.action, transition.reward)) for transition in list(self.memory)[:x]]

    
    def last_element(self):
        return list(self.memory)[-1]

    def __len__(self):
        return len(self.memory)
    
    


class PrioritizedReplayMemory2(object):
    """
    INSPO OF "LEVERAGIING DEMOSNTARTYIONS FOR DEEP RL ON ROBOTICS PROBLEMS WITH SPARSE REWARDS"
    """
    def __init__(self, capacity, alpha=0.3, beta=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.size = 0
        self.max_priority = 1e6

    def push(self, *args):
        
        self.memory.append(Transition(*args))
        self.priorities.append(self.max_priority)
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities / np.sum(priorities)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        transitions = [self.memory[idx] for idx in indices]
        
        
        return transitions, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
           self.priorities[idx] = priority
           self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return self.size

    
class PrioritizedReplayMemory(object):
    '''
    Replay memory with priority, the priority is set by losses. Each element of the memory is assigned a very high priority a priori, 
    so that these samples are chosen when sampling the memory. Once a batch is established, i.e. the memory is sampled, 
    the memory weights are updated with the losses of each of the elements that make up the batch.
    
    '''

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.priorities = deque([], maxlen=capacity)
        self.cont = 0
        self.cont_batch=0
    def push(self, *args):
        """Guarda una transición con su prioridad."""
        self.memory.append(Transition(*args))
        # A cada elemento de la memoria guarado se le asigna una prioridad inicial alta, por ejemplo 10000 (la idea es que sea 
        # bastante más alto que las prioridades que se estiman de las perdidas para que sean elegidos)
        self.priorities.append(1000000)
    def __len__(self):
        return len(self.memory)
    def sample(self, batch_size):
        """Realiza un muestreo basado en prioridades."""
        total_priority = sum(self.priorities)
        probabilities = [p / total_priority for p in self.priorities]
        
        for prob in probabilities:
            if prob <0:
                pdb.set_trace()
        # Indices del batch
        sampled_indices = random.choices(range(len(self.memory)), k=batch_size, weights=probabilities)
        
        # Indices sin duplicados
        unique_indices = list(set(sampled_indices))
        
        # Mientras haya duplicados
        while len(unique_indices) != batch_size:
            # filtramos indices de manera que nos quedamos solo con los que no estén ya seleccionados
            posible_indexes = [i for i in range(len(self.memory)) if i not in unique_indices]
            # pdb.set_trace()
            posible_probabilities = [probabilities[i] for i in posible_indexes]
            # numero de indices que se necesitan para completar el batch
            k_len = batch_size - len(unique_indices)
            sample_indices = random.choices(posible_indexes, k=k_len, weights=posible_probabilities)
            for index in sample_indices:
                unique_indices.append(index)
            unique_indices = list(set(unique_indices))
       
        transitions = [self.memory[i] for i in unique_indices]

        return transitions, unique_indices

    def update_priorities(self, indices, priorities):
        """Actualiza las prioridades de las transiciones muestreadas."""
        for i, priority in zip(indices, priorities):
            self.priorities[i] = priority
        # print(indices)
        # print(self.priorities)
        # print(np.max(self.priorities))
        
    def plot_priorities(self, save_path):
        """Dibuja un gráfico de barras de las prioridades."""
        self.cont += 1
        if self.cont % 200 == 0:
            # priorities = list(self.priorities)
            total_priority = sum(self.priorities)
            probabilities = [p / total_priority for p in self.priorities]
            fig, ax = plt.subplots(figsize=(10, 5))
            # Dibuja un gráfico de barras
            ax.bar(range(len(probabilities)), probabilities, alpha=0.7)
            
            # Dibuja líneas que conectan los puntos
            # for i in range(len(index)):
            #     ax.plot([index[i], index[i]], [0, priorities[i]], 'b-')
            
            # Marca los puntos en el gráfico
            ax.scatter(range(len(probabilities)), probabilities, c='r', marker='o')
            
            # Alinea los ejes x e y en el punto (0, 0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            plt.xlabel('Memory index')
            plt.ylabel('Priority')
            plt.title('Priority of transitions')
            fig.savefig(save_path+'/Priorities_replay_memory_'+str(self.cont)+'.jpg')
        if self.cont % 100 == 0:
            # priorities = list(self.priorities)
            total_priority = sum(self.priorities)
            probabilities = [p / total_priority for p in self.priorities]
            fig, ax = plt.subplots(figsize=(10, 5))
            # Dibuja un gráfico de barras
            ax.bar(range(len(probabilities)-1), probabilities[:-1], alpha=0.7)
            
            # Dibuja líneas que conectan los puntos
            # for i in range(len(index)):
            #     ax.plot([index[i], index[i]], [0, priorities[i]], 'b-')
            
            # Marca los puntos en el gráfico
            ax.scatter(range(len(probabilities)-1), probabilities[:-1], c='r', marker='o')
            
            # Alinea los ejes x e y en el punto (0, 0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            plt.xlabel('Memory index')
            plt.ylabel('Priority')
            plt.title('Priority of transitions')
            fig.savefig(save_path+'/Priorities_replay_memory_without_last_'+str(self.cont)+'.jpg')
    def plot_batch_priorities(self, index, save_path):
        """Dibuja un gráfico de barras de las prioridades."""
        self.cont_batch += 1
        if self.cont_batch % 200 == 0:
            total_priority = sum(self.priorities)
            probabilities = [p / total_priority for p in self.priorities]
            probabilities = [probabilities[i] for i in index]
            fig, ax = plt.subplots(figsize=(10, 5))
    
            # Dibuja un gráfico de barras
            ax.bar(index, probabilities, alpha=0.7)
            
            # # Dibuja líneas que conectan los puntos
            # for i in range(len(index)):
            #     ax.plot([index[i], index[i]], [0, probabilities[i]], 'b-')
            
            # Marca los puntos en el gráfico
            ax.scatter(index, probabilities, c='r', marker='o')
            
            # Alinea los ejes x e y en el punto (0, 0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            plt.xlabel('Memory index')
            plt.ylabel('Priority')
            plt.title('Priority of transitions in the batch')
            fig.savefig(save_path+'/Priorities_replay_memory_in_batch_'+str(self.cont_batch)+'.jpg')
        if self.cont_batch % 100 == 0:
            total_priority = sum(self.priorities)
            
            probabilities = [p / total_priority for p in self.priorities]
            index = sorted(index)
            probabilities = [probabilities[i] for i in index]
            # pdb.set_trace()
            
            fig, ax = plt.subplots(figsize=(10, 5))
    
            # Dibuja un gráfico de barras
            ax.bar(index[:-1], probabilities[:-1], alpha=0.7)
            
            # # Dibuja líneas que conectan los puntos
            # for i in range(len(index)-1):
            #     ax.plot([index[:-1][i], index[:-1][i]], [0, probabilities[:-1][i]], 'b-')
            
            # Marca los puntos en el gráfico
            ax.scatter(index[:-1], probabilities[:-1], c='r', marker='o')
            
            # Alinea los ejes x e y en el punto (0, 0)
            ax.spines['left'].set_position('zero')
            ax.spines['bottom'].set_position('zero')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
            plt.xlabel('Memory index')
            plt.ylabel('Priority')
            plt.title('Priority of transitions in the batch')
            fig.savefig(save_path+'/Priorities_replay_memory_in_batch_without_last_'+str(self.cont_batch)+'.jpg')
            
            
            
            
if __name__ == '__main__':

    t1 = Transition(-1, 0, 'a', -2, 1, -1)
    t2 = Transition(-2, 1, 'b', -2, 2, -1)
    t3 = Transition(-3, 1, 'c', -3, 2,  0)

    rm = ReplayMemory(3)
    rm.push(*t1)
    rm.push(*t2)
    rm.push(*t3)

    # now each transition is there
    print(rm.sample(3))

    # now this push has removed t1 from the replay memory
    rm.push(*t3)
    print(rm.sample(3))


