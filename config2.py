import torch 

# ENVIROMENT CONFIG

# DQNs config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training config
LR = 1e-4
EPOCHS = 1
NUM_EPISODES = 300
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
N_STEP = 1

# REPLAY MEMORY
MEMORY_SIZE = 10000