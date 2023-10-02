"""
This is the code that needs to be launched to train the model
"""
import math
import gym
from itertools import count
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DQN2 import DQN_MLP
from statTracker import StatTracker
from memory import ReplayMemory, Transition
from plotUtils import plotEpisodeStats, plotEpochStats

import torch
import torch.optim as optim
import torch.nn as nn


def selectAction(state, exp):


    """
    This function implements and e-greedy policy for action taking
    """
    cfg = exp['cfg']
    env = exp['env']
    policyNet = exp['policyNet']

    # 1 - Compute the epsilonTh and update the stepsDone variable
    epsTh = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
        math.exp(-1. * exp['stepsDone'] / cfg.EPS_DECAY)
    exp['stepsDone'] += 1

    # 2 - with probability epsTh we take the random action
    r = random.random()
    if( r < epsTh ):
        randomAction = env.action_space.sample()
        action = torch.tensor([[randomAction]], dtype=torch.long, device=cfg.DEVICE)

    # otherwise, optimal action according to policyNet
    else:
        with torch.no_grad():
            action = policyNet(state).max(1)[1].view(1,1)

    return action, epsTh

def nStep(exp, action):

    """
    The enviroment is in a state and we are going to take n-actions (n-env.steps())
    If we want to implement an n-step, the first action is the one passed as
    argument to this function (the eps-greedy-possibly-random action). After the
    action passed as an argument, the next actions taken up to the n-step should
    all be following optimal policy. This is, using max_a(policyNet(newStates)).
    Once we had taken n-step actions, we can acumulate the discounted-reward 
    in the reward variable, and set as next state the ending state given back
    by the enviroment. The rest of the code outside this function shouldn't 
    change. With this approach we will take n-actions in a single time-step of
    the outside for t in count() loop for the episode. So episodes are likely to
    finish earlier and less updates to the models will be done as n-increases.

    If before finishing the n-steps we reach a terminal state, then, that particular
    entry por the memory won't be n-step long.
    """
    env = exp['env']

    observation, reward, terminated, truncated, _ = env.step(action)
    reward = torch.tensor([reward], device=cfg.DEVICE)
    done = terminated or truncated

    if terminated:
        nextState = None
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

    return (nextState, reward, done)

def optimizeModel( exp ):

    """
    In this method we are going to compute two things:
        - The target values for each transition in the replay memory
        - The current estimation of our model for Q(St,At)
    
    Note that the taget values Q(St+n,a) should be zero for entries
    in the memory whose next_state = None. This is because after taking
    the action Q(St,a) the episode ends, and no more future rewards will
    follow. So the target is: Rt+1 + 0 (i.e., the Q(St,At) because that
    particular action At at state St has led to a terminal state, you
    only get the Rt+1 and nothign else to account for the future). Thus
    max_a(Q(St+1),a) = 0.
    """

    rm = exp['replayMemory']
    cfg = exp['cfg']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']
    criterion = exp['criterion']
    optimizer = exp['optimizer']


    # 1 - Do nothing if the memory still don't have BATCH_SIZE samples
    if( len(rm) < cfg.BATCH_SIZE ):
        return 0, 0
     
    # 2 - Sample a batch of n-step transitions
    transitions = rm.sample(cfg.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 3 - We need to create batches as input for the models. The problem is 
    # that when creating the nextState batch there might be None's. Thus, 
    # we need some mask to avoid passing those to the target network
    nonFinalMask = torch.tensor( 
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=cfg.DEVICE, dtype=torch.bool)
    
    nextStateBatch = torch.cat([s for s in batch.next_state if s is not None])
    stateBatch     = torch.cat(batch.state)
    actionBatch    = torch.cat(batch.action)
    rewardBatch    = torch.cat(batch.reward)

    # 4 - Use St,At to get the current estimation for Q(St,At) that the 
    # policyNet model is currently giving
    out = policyNet(stateBatch)
    stateActionValues = out.gather(1, actionBatch)
    avgMaxQ = out.max(1)[0].mean()
    
    # 5 - Compute the state action value term of the target equation. Note 
    # that we have to take max_a{(Qt+n,a)}
    nextStateActionValues = torch.zeros(cfg.BATCH_SIZE, device=cfg.DEVICE)
    with torch.no_grad():
        nextStateActionValues[nonFinalMask] = targetNet(nextStateBatch).max(1)[0]

    # 6 - Now we can compute the targets for the loss function. Here we have to
    # take care of the n-step discounts
    expectedStateActionValues = (cfg.GAMMA * nextStateActionValues) + rewardBatch
    
    # 7 - Finally, let's compute the loss and update the model
    loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policyNet.parameters(), 100)
    optimizer.step()

    return loss.item(), avgMaxQ.item()

def updateTargetNet( exp ):

    """
    This method is in charge of how and when update the targetNet
    """    
    targetNet = exp['targetNet']
    policyNet = exp['policyNet']
    TAU = exp['cfg'].TAU

    targetStateDict = targetNet.state_dict()
    policyStateDict = policyNet.state_dict()

    # soft update implementation
    for key in policyStateDict:
        targetStateDict[key] = policyStateDict[key]*TAU + targetStateDict[key]*(1-TAU)

    # other options...

    # finally, update the targetNet weights
    targetNet.load_state_dict(targetStateDict)
    return

def train(epoch, exp):

    env = exp['env']
    cfg = exp['cfg']
    rm  = exp['replayMemory']
    st  = exp['statTracker']

    for epi in range(cfg.NUM_EPISODES):

        # 1 - Get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

        # for each time step until end of episodeoptimizeModel
        for t in count():

            # 3.1 select an action based on current state and policy
            action, epsTh = selectAction(state, exp)

            # 3.2 this is the section in charge of generating entries for the replayMemory
            # Here is where you can implement n-step strategies
            (nextState, reward, done) = nStep(exp, action.item())

            # 3.3 store this new transition in memory
            rm.push(state, action, nextState, reward)

            # 3.4 move to the next state
            state = nextState

            # 3.5 perform policyNet learning (model optimization)
            loss,avgMaxQ = optimizeModel(exp)

            # 3.6 update the targetNet
            updateTargetNet(exp)

            # 3.7 track some useful stats
            st.add(epoch, epi,t,'epsilon', epsTh)
            st.add(epoch, epi,t,'batch_loss', loss)
            st.add(epoch, epi,t,'batch_avgMaxQ', avgMaxQ)
            st.add(epoch, epi,t,'instant_reward', reward.item())

            # 3.8 check if the episode is finished
            if done:
                st.add(epoch,epi,t,'episode_durations', t)
                print("Episode %d, duration %d" %(epi, t))
                break
        
        # the episode is finished, let's plot stuff
        #plotEpisodeInfo(st, epoch, epi)

    plotEpochStats(st, epoch)
    return

def validation( epoch, exp):

    return

def loadConfiguration():
    
    print("Loading config...")
    """
    Careful here! the enviroment is using the other config! config.py instead of config2.py
    """
    import config2 as cfg
    
    print("\tDone!")
    return cfg

def initExperiment(cfg):
    
    print("Initializing experiment variables...")

    # 1 - Initialize enviroment
    #env = gym.make("gym_basic:basic-v0", display = cfg.ENV_DISPLAY, disable_env_checker = cfg.DISABLE_ENV_CHECKER)
    env = gym.make('CartPole-v1')
    num_actions = env.action_space.n
    state, info = env.reset()
    state_dimensionality = len(state)
    stepsDone = 0

    # 2 - Initialize DQN networks, optimizer and loss function
    policyNet = DQN_MLP(state_dimensionality, num_actions).to(cfg.DEVICE)
    targetNet = DQN_MLP(state_dimensionality, num_actions).to(cfg.DEVICE)
    targetNet.load_state_dict(policyNet.state_dict())

    optimizer = optim.AdamW(policyNet.parameters(), lr = cfg.LR, amsgrad=True) 
    criterion = nn.SmoothL1Loss()
    
    # 3 - Initialize stat tracking objects
    st = StatTracker()

    # 4 - Initialize replay memory
    rm = ReplayMemory(cfg.MEMORY_SIZE)

    exp = {'cfg': cfg,
           'env': env,
           'stepsDone': stepsDone,
           'policyNet': policyNet,
           'targetNet': targetNet,
           'optimizer': optimizer,
           'criterion': criterion,
           'statTracker': st,
           'replayMemory': rm}

    print("\tDone!")
    return exp

def runExperiment(exp):

    print("Starting experiment")
    for e in range(cfg.EPOCHS):
        train(e, exp)
        validation(e, exp)

    print("\tDone!")
    return

def saveAndVisualize(exp):
    
    print("Saving and visualizing results")
    
    print("\tDone!")
    return

if( __name__ == '__main__' ):

    # 1 - Load the experiment config
    cfg = loadConfiguration()

    # 2 - Init experiment env, models, opt, loss, stat objects, etc.
    exp = initExperiment(cfg)

    # 3 - Run the training/validation process
    runExperiment(exp)

    # 4 - Save and visualize results
    saveAndVisualize(exp)