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

import config2 as cfg

class DQN_LSTM_LateFusion(nn.Module):

    def __init__(self, input_size,hidden_size_LSTM, output_size):
        super(DQN_LSTM_LateFusion, self).__init__()

        self.input_size = input_size
        self.hidden_size_LSTM = hidden_size_LSTM
        self.output_size = output_size

        self.embedding_size = 32
        self.resetHidden()


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
        
        self.hidden1 = nn.Sequential(
            nn.Linear(self.embedding_size*4, 256),
            # nn.BatchNorm1d(512),
            nn.ReLU()
        )


        self.lstm = nn.LSTM(256, hidden_size_LSTM, batch_first = True)
        self.init_lstm_weights()
        #self.norm = torch.nn.LayerNorm(hidden_size_LSTM)

        # self.init_lstm_weights()
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size_LSTM, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )       
       
    def resetHidden(self):

        self.hidden = (torch.zeros(1,self.hidden_size_LSTM).to(cfg.DEVICE),
                       torch.zeros(1,self.hidden_size_LSTM).to(cfg.DEVICE))
        self.prevHidden = (torch.zeros(1,self.hidden_size_LSTM).to(cfg.DEVICE),
                           torch.zeros(1,self.hidden_size_LSTM).to(cfg.DEVICE))
        return
    
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
                
    def forward(self, inputs, hidden=None):

        hiddenT = self.hidden if hidden is None else hidden

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

        x = torch.cat((x1, x2, x3, x4), dim)         
        x = self.hidden1(x)
        
        output, (hx, cx) = self.lstm(x, hiddenT)
        #x = self.norm(output)
        x = self.output_layer(output)

        if hidden is None:
            self.prevHidden = (self.hidden[0].clone(), self.hidden[1].clone())
            self.hidden = (hx,cx)

        return x, (hx,cx) 
    

class DQN_MLP(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN_MLP, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

        # hidden and prevHidden are not used in this model, they are here just
        # for compatibility reasons with the rest of the code
        self.hidden = (torch.zeros(1,1).to(cfg.DEVICE),
                       torch.zeros(1,1).to(cfg.DEVICE))
        self.prevHidden = self.hidden
        return

    def resetHidden(self):

        self.hidden = (torch.zeros(1,1).to(cfg.DEVICE),
                       torch.zeros(1,1).to(cfg.DEVICE))
        return
    
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    # hidden is only here for compatibility reasons
    def forward(self, x, hidden=None):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x), self.hidden


if __name__ == '__main__':

    model = DQN_LSTM_LateFusion(133, hidden_size_LSTM=512, output_size=6).to(cfg.DEVICE)
    
    batch_size = 1
    sequence_length = 1
    input_size = 133

    x1 = torch.randn(batch_size, input_size).to(cfg.DEVICE)
    x2 = torch.randn(batch_size, input_size).to(cfg.DEVICE)
    x3 = torch.randn(batch_size, input_size).to(cfg.DEVICE)

    out, hidden = model(x1)
    out, hidden = model(x2)
    out, hidden = model(x3)
    a = 3
