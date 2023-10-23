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

TEMPORAL_CTX = cfg.ACTION_SPACE +1

class DQN_LSTM_LateFusion(nn.Module):
    def __init__(self, input_size,hidden_size_LSTM, output_size):
        super(DQN_LSTM_LateFusion, self).__init__()

        self.embedding_size = 32
                       
        self.input_AcPred = nn.Sequential(
            nn.Linear(33, self.embedding_size),
            # nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()
        )
        
        self.input_AcRec = nn.Sequential(
            nn.Linear(33, self.embedding_size),
            # nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_VWM = nn.Sequential(
            nn.Linear(44, self.embedding_size),
            # nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        self.input_OiT = nn.Sequential(
            nn.Linear(23, self.embedding_size),
            # nn.BatchNorm1d(self.embedding_size),
            nn.ReLU()            
        )
        
        if cfg.TEMPORAL_CONTEXT:
            self.input_TempCtx = nn.Sequential(
                nn.Linear(7, self.embedding_size),
                # nn.BatchNorm1d(self.embedding_size),
                nn.ReLU()            
            )
        
        self.hidden1 = nn.Sequential(
            nn.Linear(self.embedding_size*4, 256),
            # nn.BatchNorm1d(512),
            nn.ReLU()
        )


        self.lstm = nn.LSTM(256, hidden_size_LSTM, batch_first = True)
        self.init_lstm_weights()
        self.norm = torch.nn.LayerNorm(hidden_size_LSTM)
        # self.lstm = nn.Sequential(
        #     nn.LSTM(512, hidden_size_LSTM, batch_first = True),
        #     nn.LayerNorm(hidden_size_LSTM)
        #     )

        # self.init_lstm_weights()
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size_LSTM, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)

        )       
       
        
    def init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
           if "weight_ih" in name:
               for sub_name in ["weight_ih_i", "weight_ih_f", "weight_ih_g", "weight_ih_o"]:
                   if sub_name in name:
                       nn.init.xavier_uniform_(param.data)
           elif "weight_hh" in name:
               for sub_name in ["weight_hh_i", "weight_hh_f", "weight_hh_g", "weight_hh_o"]:
                   if sub_name in name:
                       nn.init.orthogonal_(param.data)
           elif "bias" in name:
               nn.init.constant_(param.data, 0)
                
    def forward(self, inputs, hidden):

        
        # Separate input tensor
        # print(hidden.shape)
        # pdb.set_trace()

        if len(inputs.shape) < 3:
  
            ac_pred = inputs[:, 0:33]
            ac_rec = inputs[:, 33:66]
            vwm = inputs[:, 66:110]
            oit = inputs[:, 110:133]
            dim = 1
        else:
            ac_pred = inputs[:,:, 0:33]
            ac_rec = inputs[:, :,33:66]
            vwm = inputs[:, :,66:110]
            oit = inputs[:, :,110:133]
            dim = 2

        # pdb.set_trace()
        x1 = self.input_AcPred(ac_pred)
        x2 = self.input_AcRec(ac_rec)
        x3 = self.input_VWM(vwm)
        x4 = self.input_OiT(oit)
        if cfg.TEMPORAL_CONTEXT:
            temp_ctx = inputs[:, :,133:140]
            x5 = self.input_TempCtx(temp_ctx)
            x = torch.cat((x1, x2, x3, x4, x5), dim) 
        else:
            x = torch.cat((x1, x2, x3, x4), dim)         
        x = self.hidden1(x)
        # x = self.hidden2(x)
        # x = self.relu(self.input_layer1(x))
        
        output, (hx, cx) = self.lstm(x, hidden)
        x = self.norm(output)
        x = self.output_layer(output)
        return x, (hx,cx) 
    
class DQN_LSTM(nn.Module):
    def __init__(self, input_size,hidden_size_LSTM, output_size):
        super(DQN_LSTM, self).__init__()
            
        if cfg.TEMPORAL_CONTEXT:
            self.feature_dim = 133 + TEMPORAL_CTX
        else:
            self.feature_dim = 133 #First features and Z variable separated
        
        self.input_layer1 = nn.Linear(self.feature_dim, 256)
        self.lstm = nn.LSTM(256, hidden_size_LSTM, batch_first = True)
        self.init_lstm_weights()
        self.norm = torch.nn.LayerNorm(hidden_size_LSTM)
        self.output_layer = nn.Linear(hidden_size_LSTM, output_size)

        self.relu = nn.ReLU()
        
        
    def init_lstm_weights(self):
        for name, param in self.lstm.named_parameters():
           if "weight_ih" in name:
               for sub_name in ["weight_ih_i", "weight_ih_f", "weight_ih_g", "weight_ih_o"]:
                   if sub_name in name:
                       nn.init.xavier_uniform_(param.data)
           elif "weight_hh" in name:
               for sub_name in ["weight_hh_i", "weight_hh_f", "weight_hh_g", "weight_hh_o"]:
                   if sub_name in name:
                       nn.init.orthogonal_(param.data)
           elif "bias" in name:
               nn.init.constant_(param.data, 0)
                
    def forward(self, inputs, hidden):

        
        # Separate input tensor
        input1 = inputs[:, 0:self.feature_dim]

        x1 = self.relu(self.input_layer1(input1))
        
        output, (hx, cx) = self.lstm(x1, hidden)
        x = self.norm(output)
        x = self.output_layer(x)
        return x, (hx,cx) 

        
class DQN(nn.Module):

    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
            
        if cfg.TEMPORAL_CONTEXT:
            self.feature_dim = 133 + TEMPORAL_CTX
        else:
            self.feature_dim = 133 #First features and Z variable separated
        
        self.input_layer1 = nn.Linear(self.feature_dim, 256)
        self.output_layer = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        
    
                
    def forward(self, inputs):

        
        # Separate input tensor
        input1 = inputs[:, 0:self.feature_dim]
        x1 = self.relu(self.input_layer1(input1))
        x = self.output_layer(x1)

        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
        

#----------------------------------        
        
if cfg.LSTM:
    if cfg.IMPORTANCE_SAMPLING:
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'hidden', 'importance_sampling'))   
    else:
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'hidden'))   
else:
    if cfg.IMPORTANCE_SAMPLING:
        Transition = namedtuple('Transition', ('state', 'action', 'reward', 'importance_sampling'))   
    else:
        Transition = namedtuple('Transition', ('state', 'action', 'reward'))   
# Transition = namedtuple('Transition', ('state', 'action', 'reward'))      
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        
        return random.sample(self.memory, batch_size)
        
    def show_batch(self, batch_size):
        """
        Prints 'batch_size' random elements stored in the memory.
        Input:
            batch_size: number of entries to be displayed.
        """
        for i in range(batch_size):
            print(random.sample(self.memory, 1), "\n")
        return 0
    
    def __len__(self):
        return len(self.memory)            



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
        self.priorities.append(10000)

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
        



        # plt.show()
        
    def __len__(self):
        return len(self.memory)
