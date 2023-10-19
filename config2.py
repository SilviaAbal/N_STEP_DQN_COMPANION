import torch 



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

# ENVIROMENT CONFIG
ROBOT_ACTIONS_MEANINGS = {
	0: 'bring butter', 1: 'bring jam', 2: 'bring milk',	3: 'bring nutella',	4: 'bring tomato sauce', 5: 'do nothing',
    }

ROBOT_ACTIONS_2_OBJECTS = {
    0: 'butter', 1: 'jam', 2: 'milk', 3: 'nutella', 4: 'tomato sauce', 5: 'do nothing',
    }

ROBOT_AVERAGE_DURATIONS = {
 	0: 174, 1: 198, 2: 186, 3: 234, 4: 270, 5: 0,
    'butter': 174, 'jam': 198, 'milk': 186, 'nutella': 234, 'tomato sauce': 270, 'do nothing': 0, 
    }


ATOMIC_ACTIONS_MEANINGS = {
	0: 'other manipulation', 1: 'pour milk', 2: 'pour water', 3: 'pour coffee', 4: 'pour Nesquik', 5: 'pour sugar',	6: 'put microwave',
	7: 'stir spoon', 8: 'extract milk fridge', 9: 'extract water fridge', 10: 'extract sliced bread', 11: 'put toaster', 12: 'extract butter fridge',
	13: 'extract jam fridge', 14: 'extract tomato sauce fridge', 15: 'extract nutella fridge', 16: 'spread butter', 17: 'spread jam',
	18: 'spread tomato sauce', 19: 'spread nutella', 20: 'pour olive oil', 21: 'put jam fridge', 22: 'put butter fridge',
	23: 'put tomato sauce fridge', 24: 'put nutella fridge', 25: 'pour milk bowl', 26: 'pour cereals bowl', 27: 'pour nesquik bowl',
	28: 'put bowl microwave', 29: 'stir spoon bowl', 30: 'put milk fridge', 31: 'put sliced bread plate', 32: 'TERMINAL STATE',
    }

OBJECTS_MEANINGS = {
	0: 'background', 1: 'bowl',	2: 'butter', 3: 'cereals', 4: 'coffee',	5: 'cup', 6: 'cutting board', 7: 'fork', 8: 'fridge',
	9: 'jam', 10: 'knife', 11: 'microwave', 12: 'milk', 13: 'nesquik', 14: 'nutella', 15: 'olive oil', 16: 'plate', 17: 'sliced bread',
	18: 'spoon', 19: 'sugar', 20: 'toaster', 21: 'tomato sauce', 22: 'water'
    }


OBJECTS_INIT_STATE = { # whether an object is recheable by the person (1) or is outside of reach (0, fridge)t
    'background': 0, 'bowl': 1, 'butter': 0, 'cereals': 1, 'coffee': 1, 'cup': 1, 'cutting board': 1, 'fork': 1, 'fridge': 1, 
    'jam': 0, 'knife': 1, 'microwave': 1, 'milk': 0, 'nesquik': 1, 'nutella': 0, 'olive oil': 1, 'plate': 1, 'sliced bread': 1,
	'spoon': 1, 'sugar': 1, 'toaster': 1, 'tomato sauce': 0, 'water': 1
    } 

ENV_VERBOSE = True
ANNOTATIONS_FOLDER = './video_annotations'
DATASET_NAME = 'dataset_pred_recog_tmp_ctx'
LABELS_FILE_NAME = 'labels_updated.pkl'
ANNOTATIONS_FPS = 5
ANNOTATIONS_DIM = 110 # the number of features from gaze
VIDEO_FPS = 30
DECISION_RATE = 30 
ROBOT_ATOMIC_ACTIONS = [8, 12, 13, 14, 15]
ASR_PENALTY = 3.5*VIDEO_FPS # 3.5 seconds expressed as frames
FACTOR_ENERGY_PENALTY = 0.1
IMPOSSIBLE_ACTION_PENALTY = 10

INTERACTIVE_OBJECTS_ROBOT = ['butter','jam','milk','nutella','tomato sauce']
NUM_ROBOT_ACTIONS = len(ROBOT_ACTIONS_MEANINGS)
NUM_OBJECTS = len(OBJECTS_MEANINGS)
