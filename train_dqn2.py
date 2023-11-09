"""
This is the code that needs to be launched to train the model
"""
import math
import os
import glob
import gym
from itertools import count
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from DQN2 import DQN_MLP, DQN_LSTM_LateFusion
from statTracker import StatTracker
from memory import ReplayMemory, Transition
from plotUtils import plotEpisodeStats, plotEpochStats, plotHelper2




def selectAction(state, exp, test=False):


    """
    This function implements and e-greedy policy for action taking
    """
    cfg = exp['cfg']
    env = exp['env']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']

    # 1 - Compute the epsilonTh
    epsTh = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
        math.exp(-1. * exp['stepsDone'] / cfg.EPS_DECAY)
    epsTh = 0 if test else epsTh

    # 2 - with probability epsTh we take the random action
    r = random.random()
    if( r < epsTh):
        randomAction = env.action_space.sample()
        action = torch.tensor([[randomAction]], dtype=torch.long, device=cfg.DEVICE)
        with torch.no_grad(): policyNet(state)

    # otherwise, optimal action according to policyNet
    else:
        with torch.no_grad():
            action = policyNet(state)[0].max(1)[1].view(1,1)

    # we have to advance the target net (in case is a lstm)
    with torch.no_grad(): targetNet(state)

    # we need to return the original hidden state of the policyNet for the replay memory
    pHidden = (policyNet.prevHidden[0].clone(), policyNet.prevHidden[1].clone())

    return action, pHidden, epsTh

def nStep(exp, action, epoch, epi, ti):

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
    cfg = exp['cfg']
    env = exp['env']
    st  = exp['statTracker']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']

    N_STEP = cfg.N_STEP
    GAMMA  = cfg.GAMMA
    
    # G is the inmediate discounted n_step reward: 
    # Rt+1 + gamma*Rt+2 + gamma^2*Rt+3 ...
    G = 0
    t = 0
    actionsTaken = {}

    # the first action is the epsilon greedy action
    actionsTaken[env.episode.currentFrame] = action
    observation, reward, terminated, truncated, _ = env.step(action)

    exp['stepsDone'] += 1
    t += 1
    G += reward
    
    # the rest of the N-step actions, have to be optimal
    with torch.no_grad():
        for n in range(1,N_STEP):

            # 1 - check if env has ended
            if terminated or truncated:
                break 

            # 2 - find the next optimal action (also advance targetNet)
            nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
            nextAction = policyNet(nextState)[0].max(1)[1].view(1,1)
            targetNet(nextState)

            # 3 - take the a new step
            actionsTaken[env.episode.currentFrame] = nextAction.item()
            observation, reward, terminated, truncated, _ = env.step(nextAction.item())

            exp['stepsDone'] += 1
            t += 1

            # 4 - update the discounted reward
            G += (GAMMA**n) * reward
        # 
    
    # check if the last n_step ended on a terminal state
    if terminated:
        nextState = None
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

    reward = torch.tensor([G], device=cfg.DEVICE)
    done = terminated or truncated
    st.add(epoch, epi, ti+t, 'actions_taken', actionsTaken)

    # we need the corresponding hidden of the targetLSTM to be store along nextState
    tHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
    return (nextState, tHidden, reward, done, t)

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

    ph0 = torch.stack([t[0] for t in batch.pHidden], dim=1)
    pc0 = torch.stack([t[1] for t in batch.pHidden], dim=1)
    th0 = torch.stack([t[0] for t in batch.tHidden], dim=1)
    tc0 = torch.stack([t[1] for t in batch.tHidden], dim=1)
    th0 = th0[nonFinalMask.unsqueeze(0)].unsqueeze(0)
    tc0 = tc0[nonFinalMask.unsqueeze(0)].unsqueeze(0)

    # 4 - Use St,At to get the current estimation for Q(St,At) that the 
    # policyNet model is currently giving
    out = policyNet(stateBatch.unsqueeze(1), (ph0, pc0))[0].squeeze(1)
    stateActionValues = out.gather(1, actionBatch)
    avgMaxQ = out.max(1)[0].mean()
    
    # 5 - Compute the state action value term of the target equation. Note 
    # that we have to take max_a{(Qt+n,a)}
    nextStateActionValues = torch.zeros(cfg.BATCH_SIZE, device=cfg.DEVICE)
    with torch.no_grad():
        tma = targetNet(nextStateBatch.unsqueeze(1), (th0, tc0))[0].squeeze(1)
        nextStateActionValues[nonFinalMask] = tma.max(1)[0]

    # 6 - Now we can compute the targets for the loss function. Here we have to
    # take care of the n-step discounts only for the model prediction (the instant rewards
    # are done in the n_step function)
    expectedStateActionValues = ( (cfg.GAMMA**cfg.N_STEP) * nextStateActionValues) + rewardBatch
    
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

def stepTest(exp, action, epoch, epi, ti, env, st):

    cfg = exp['cfg']
    
    # G is the inmediate discounted n_step reward: 
    # Rt+1 + gamma*Rt+2 + gamma^2*Rt+3 ...
    G = 0
    t = 0
    actionsTaken = {}

    # take the action
    actionsTaken[env.episode.currentFrame] = action
    observation, reward, terminated, truncated, _ = env.step(action)

    exp['stepsDone'] += 1
    t += 1
    G += reward
    
    # check if the last n_step ended on a terminal state
    if terminated:
        nextState = None
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

    done = terminated or truncated
    st.add(epoch, epi, ti+t, 'actions_taken', actionsTaken)

    return (nextState, reward, done, t)

def train(epoch, exp):

    env = exp['env']
    st  = exp['statTracker']
    cfg = exp['cfg']
    rm  = exp['replayMemory']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']

    for epi in range(cfg.NUM_EPISODES):

        # 0 - reset the initial hidden states of the models
        policyNet.resetHidden()
        targetNet.resetHidden()

        # 1 - Get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

        # for each time step until end of episodeoptimizeModel
        t = 0
        while True:

            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp)

            # 3.2 this is the section in charge of generating entries for the replayMemory
            # Here is where you can implement n-step strategies
            (nextState, tHidden, reward, done, stepsTaken) = nStep(exp, action.item(), epoch, epi, t)
            t += stepsTaken

            # 3.3 store this new transition in memory
            rm.push(pHidden, state, action, tHidden, nextState, reward)

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
            st.add(epoch, epi,t,'episode_durations', t)
            st.add(epoch, epi,t,'episode_delay', env.episode.delay)
            st.add(epoch, epi,t,'episode_name', env.episode.folderName)
            st.add(epoch, epi,t,'reactive_time', env.episode.reactiveTime)
            st.add(epoch, epi,t,'min_delay', env.episode.minDelay)

            # 3.8 check if the episode is finished
            if done:    
                print("Episode %d, delay %d" %(epi, env.episode.delay))
                env.episode.storeActionsTaken(epoch, epi, st)
                break
        
        # the episode is finished, let's plot stuff
        # plotEpisodeStats(st, epoch, epi)

    plotEpochStats(st, epoch)
    return

def test( epoch, exp, env, st):

    cfg = exp['cfg']
    policyNet = exp['policyNet']

    for epi in range(env.numEpisodes):

        # 0 - reset the initial hidden states of the models
        policyNet.resetHidden()

        # 1 - Get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

        # for each time step until end of episode
        t = 0
        while True:

            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp, test=True)

            # 3.2 take the step
            (nextState, reward, done, stepsTaken) = stepTest(exp, action.item(), epoch, epi, t, env, st)
            t += stepsTaken

            # 3.4 move to the next state
            state = nextState

            # 3.7 track some useful stats
            st.add(epoch, epi,t,'epsilon', epsTh)
            st.add(epoch, epi,t,'batch_loss', 0)
            st.add(epoch, epi,t,'batch_avgMaxQ', 0)
            st.add(epoch, epi,t,'instant_reward', reward)
            st.add(epoch, epi,t,'episode_durations', t)
            st.add(epoch, epi,t,'episode_delay', env.episode.delay)
            st.add(epoch, epi,t,'episode_name', env.episode.folderName)
            st.add(epoch, epi,t,'reactive_time', env.episode.reactiveTime)
            st.add(epoch, epi,t,'min_delay', env.episode.minDelay)

            # 3.8 check if the episode is finished
            if done:    
                print("Episode %d, delay %d" %(epi, env.episode.delay))
                env.episode.storeActionsTaken(epoch, epi, st)
                break

    return

def loadConfiguration():
    
    print("Loading config...")
    import config2 as cfg
    
    print("\tDone!")
    return cfg

def initExperimentBackup(cfg):
    
    print("Initializing experiment variables...")

    
    # 1 - Initialize enviroment
    #env = gym.make("gym_basic:basic-v0", display = cfg.ENV_DISPLAY, disable_env_checker = cfg.DISABLE_ENV_CHECKER)
    env = gym.make("gym_basic:basic-v1", mode='train', testFile='fold1_test.txt')
    #env = gym.make('CartPole-v1')
    num_actions = env.action_space.n
    state, info = env.reset()
    state_dimensionality = len(state)
    stepsDone = 0

    # 2 - Initialize DQN networks, optimizer and loss function
    policyNet = DQN_LSTM_LateFusion(state_dimensionality, cfg.HIDDEN_SIZE_LSTM, num_actions).to(cfg.DEVICE)
    targetNet = DQN_LSTM_LateFusion(state_dimensionality, cfg.HIDDEN_SIZE_LSTM, num_actions).to(cfg.DEVICE)
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

def initExperiment(cfg, fold):
    
    print("Initializing experiment variables...")

    # 1 - Initialize enviroment
    env = gym.make("gym_basic:basic-v1", mode='train', testFile=fold)
    envVal = gym.make("gym_basic:basic-v1", mode='train', testFile=fold)
    envTest = gym.make("gym_basic:basic-v1", mode='test', testFile=fold)

    num_actions = env.action_space.n
    state, info = env.reset()
    state_dimensionality = len(state)
    stepsDone = 0

    # 2 - Initialize DQN networks, optimizer and loss function
    policyNet = DQN_LSTM_LateFusion(state_dimensionality, cfg.HIDDEN_SIZE_LSTM, num_actions).to(cfg.DEVICE)
    targetNet = DQN_LSTM_LateFusion(state_dimensionality, cfg.HIDDEN_SIZE_LSTM, num_actions).to(cfg.DEVICE)

    #policyNet = DQN_MLP(state_dimensionality,  num_actions).to(cfg.DEVICE)
    #targetNet = DQN_MLP(state_dimensionality, num_actions).to(cfg.DEVICE)

    targetNet.load_state_dict(policyNet.state_dict())

    optimizer = optim.AdamW(policyNet.parameters(), lr = cfg.LR, amsgrad=True) 
    criterion = nn.SmoothL1Loss()
    
    # 3 - Initialize stat tracking objects
    st = StatTracker()
    stVal = StatTracker()
    stTest = StatTracker()

    # 4 - Initialize replay memory
    rm = ReplayMemory(cfg.MEMORY_SIZE)

    exp = {'cfg': cfg,
           'fold': fold,
           'env': env,
           'envVal': envVal,
           'envTest': envTest,
           'stepsDone': stepsDone,
           'policyNet': policyNet,
           'targetNet': targetNet,
           'optimizer': optimizer,
           'criterion': criterion,
           'statTracker': st,
           'statTrackerVal': stVal,
           'statTrackerTest': stTest,
           'replayMemory': rm
    }
           
    print("\tDone!")
    return exp

def runExperiment(cfg):

    exp = {}

    for i,fold in enumerate(cfg.FOLDS):
        
        print("Starting experiment: %s" %(fold))

        # 1 - Init experiment env, models, opt, loss, stat objects, etc
        exp_i = initExperiment(cfg, fold)
        
        # train
        train(0, exp_i)

        # test the model in both train/test partitons (no exploration)
        test(0, exp_i, exp_i['envVal'], exp_i['statTrackerVal'])
        test(0, exp_i, exp_i['envTest'], exp_i['statTrackerTest'])
        exp[i+1] = exp_i

    print("\tDone!")
    return exp

def saveAndVisualize(cfg, exp):
    
    print("Saving and visualizing results")
    
    # 1 - Get all recipe names alfabetically sorted
    rewardVal, rewardTest, recipeNames = _recoverKFoldStat(cfg, exp, 'instant_reward', func='sum', reduction=True)
    aTakenVal, aTakenTest, _ = _recoverKFoldStat(cfg, exp, 'actions_taken', func='notIdleActions', reduction=True)
    delayVal, delayTest, _ = _recoverKFoldStat(cfg, exp, 'episode_delay', func='last', reduction=True)
    reactiveTimes, _, _ = _recoverKFoldStat(cfg, exp, 'reactive_time', func='last', reduction=True)
    minDelays, _, _ = _recoverKFoldStat(cfg, exp, 'min_delay', func='last', reduction=True)

    plotHelper2(rewardVal, rewardTest, '#recipe', 'reward', 'total reward per recipe', 
                ['val', 'test'], recipeNames, 'figs/allFolds/total_reward.png')
    plotHelper2(aTakenVal, aTakenTest, '#recipe', '#actions', 'total actions taken (!=5) per recipe', 
                ['val', 'test'], recipeNames, 'figs/allFolds/actions_taken.png')
    plotHelper2(delayVal, delayTest, '#recipe', 'delay (frames)', 'total delay per per recipe', 
                ['val', 'test', 'reacTime', 'oracle'], recipeNames, 'figs/allFolds/delay.png', reactiveTimes, minDelays)
    
    # save all the recipes (test) to visualize the kind of actions the robot took
    for k in exp.keys():
        env = exp[k]['envTest']
        for i in range(env.numEpisodes):
            epi = next(env.allEpisodes)
            epi.visualizeAndSave(mode='frames')

    print("\tDone!")
    return

def _recoverKFoldStat(cfg, exp, statName, func='sum', reduction=True):

    # 1 - Get all recipe names alfabetically sorted
    path = os.path.join(cfg.ANNOTATIONS_FOLDER, cfg.DATASET_NAME)
    recipes = sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])
    r2c = {r:i for i,r in enumerate(recipes)}
    c2r = {r:i for r,i in enumerate(recipes)}

    # 2 - Create a (folds,numRecipes) matrix for the validation stats
    # and a (1, numRecipes) vector for the test stats 
    valMat  = np.full((len(cfg.FOLDS), len(recipes)), np.nan)
    testMat = np.full((len(recipes)), np.nan)

    # 3 - Now we have to go through each fold
    for i,k in enumerate(exp.keys()):

        exp_i = exp[k]
        stVal = exp_i['statTrackerVal']
        stTest = exp_i['statTrackerTest'] 

        # get the resultVal and resultTest coordinates for the values
        valCols  = [r2c[r[0]] for r in stVal.get('episode_name', 0)]
        testCols = [r2c[r[0]] for r in stTest.get('episode_name', 0)]
        valRows  = [i] * len(valCols)

        valuesVal  = [_applyFunc(epi, func) for epi in stVal.get(statName, 0)]
        valuesTest = [_applyFunc(epi, func) for epi in stTest.get(statName, 0)]

        # lets fill the valMat and testMat
        valMat[valRows, valCols] = valuesVal
        testMat[testCols] = valuesTest

    if reduction is not None:
        valMat = np.nanmean(valMat, axis=0)

    return valMat, testMat,recipes

def _applyFunc( epi, func ):

    if func == 'sum':
        return sum(epi)

    if func == 'last':
        return epi[-1]

    if func == 'avg':
        return 1.0*sum(epi)/len(epi)
    
    if func == 'notIdleActions':
        
        epiActions = [e for t in epi for e in list(t.values())]
        return sum(1 for e in epiActions if e != 5)
    
    raise ValueError("Unsuported func in _applyFunc() method")


if( __name__ == '__main__' ):

    a = 3
    
    # 1 - Load the experiment config
    cfg = loadConfiguration()

    # 2 - Init experiment env, models, opt, loss, stat objects, etc.
    #exp = initExperiment(cfg)

    # 3 - Run the training/validation process
    exp = runExperiment(cfg)

    # 4 - Save and visualize results
    saveAndVisualize(cfg, exp)