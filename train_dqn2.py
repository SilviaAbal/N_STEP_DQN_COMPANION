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
from random import Random
from math import ldexp
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from DQN2 import DQN_MLP, DQN_LSTM_LateFusion
from statTracker import StatTracker
from memory import ReplayMemory, Transition, PrioritizedReplayMemory, PrioritizedReplayMemory2
from plotUtils import plotEpisodeStats, plotEpochStats, plotHelper2, plotHelper3
import pdb 
from datetime import datetime
import pandas as pd
import re
from procces_csv_test import proccessCsv,plotActions,visualizeSecActions
import pickle
import copy 

class rangeRandom(Random):
    """
    Example "random" number generator that provides numbers from a uniform sequence
    """
    def __init__(self, resolution=1000):
        self._pos = 0
        self._resolution = resolution

    def random_float(self):
        """
       Generate a random float using a method based on bit manipulation.
       """
        mantissa = 0x10_0000_0000_0000 | self.getrandbits(52)
        exponent = -53
        x = 0
        while not x:
            x = self.getrandbits(32)
            exponent += x.bit_length() - 32
        return ldexp(mantissa, exponent)
    def getstate(self):
        return self._pos, self._resolution

    def setstate(self, state) -> None:
        self._pos,self._resolution = state

    def seed(self, a) -> None:
        self._pos = int(a) % self._resolution
        
        
def calculatePkValues(priorities, alpha):
    """
    Calculate priority values for given priorities and alpha.
    """
    pk_values = np.sum(priorities ** alpha)
    return pk_values

def calculatePandW(pi_values, pk, alpha, beta, N):
    """
    Calculate P and w values for given pi values, pk, alpha, beta, and N.
    """
    P_values = np.array([pi ** alpha / pk for pi in pi_values])
    w_values = np.array([((1/N) * (1/P)) ** beta for P in P_values])
    return  w_values

def postProcessedPossibleActions(out,index_posible_actions):
    """
    Function that performs a post-processing of the neural network output. 
    In case the output is an action that is not available, either because 
    of the object missing or left on the table, the most likely possible action will be selected 
    from the output of the neural network,
    Parameters
    ----------
    out : (tensor)
        DQN output.
    index_posible_actions : (list)
        Posible actions taken by the robot according to the objects available.
    Returns
    -------
    (tensor)
        Action to be performed by the robot.
    """
    action_pre_processed = out.max(1)[1].view(1,1)

    if action_pre_processed.item() in index_posible_actions:
        return action_pre_processed
    else:
        out = out.cpu().numpy()
        out = out[0]
        idx = np.argmax(out[index_posible_actions])
        action = index_posible_actions[idx]
        return torch.tensor([[action]], device=cfg.DEVICE, dtype=torch.long)

    
def selectAction(state, exp, customRandom, test=False):


    """
    This function implements and e-greedy policy for action taking
    """
    cfg = exp['cfg']
    env = exp['env']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']

    # 1 - Compute the epsilonTh
    # epsTh = cfg.EPS_END + (cfg.EPS_START - cfg.EPS_END) * \
        # math.exp(-1. * exp['stepsDone'] / cfg.EPS_DECAY)
        
    epsTh = max(cfg.EPS_END, cfg.EPS_START - (cfg.EPS_START - cfg.EPS_END) * (exp['stepsDone'] / cfg.EPS_DECAY))
    # eps_threshold =max(EPS_END, EPS_START - (EPS_START - EPS_END) * (steps_done / EPS_DECAY))
    epsTh = 0 if test else epsTh

    # 2 - with probability epsTh we take the random action

    r = customRandom.random_float()

    # if NO_IMPOSSIBLE_ACTIONS is True, the system cannot perfom actions that were previouly taken in the same episode 
    # (current actions only involve taking elements from the fridge, so whenever a object is taken it is impossible to take it again in the same episode)
    if cfg.NO_IMPOSSIBLE_ACTIONS:
        objects_in_table = copy.deepcopy(env.oit)
        manipulable_objects = list(cfg.ROBOT_ACTIONS_2_OBJECTS.values())[:-1]
        mani_obj_table = [objects_in_table[obj] for obj in manipulable_objects]
        mani_obj_table.append(0) # la ultima accion es no hacer nada y siempre se puede hacer
        index_posible_actions = [i for i, x in enumerate(mani_obj_table) if x == 0]

        
    if( r < epsTh):
        if cfg.PROB_APRIORI_v1 != 0: # with labels
            index_actions = [12,13,8,15,14,0]
            probs = [env.episode.frameFeatures[env.episode.currentFrame][idx] for idx in index_actions]
            randomAction = np.argmax(probs)
            
        # if NO_ACTION_PROBABILITY = True the selection of an action in the exploration phase is not totally random, 
        # it is given more chance to select no action than actions
        if cfg.NO_ACTION_PROBABILITY != 0:
            index_posible_actions = list(range(0,env.action_space.n))
            index_no_action = index_posible_actions.index(5)
            weights = [10]*len(index_posible_actions)
            weights[index_no_action] = cfg.NO_ACTION_PROBABILITY
            # print(weights)
            randomAction = random.choices(index_posible_actions, weights, k=1)[0]
        else:
            if cfg.NO_IMPOSSIBLE_ACTIONS:
                index_action = random.randrange(len(index_posible_actions))
                randomAction = index_posible_actions[index_action]
            else:
                randomAction = env.action_space.sample()
        action = torch.tensor([[randomAction]], dtype=torch.long, device=cfg.DEVICE)
        with torch.no_grad(): policyNet(state) #####################################################

    # otherwise, optimal action according to policyNet
    else:
        with torch.no_grad():
            out = policyNet(state)[0]
            if cfg.NO_IMPOSSIBLE_ACTIONS:
                action = postProcessedPossibleActions(out,index_posible_actions)
            else:
                action = out.max(1)[1].view(1,1)
                

    # # we have to advance the target net (in case is a lstm)
    with torch.no_grad(): targetNet(state) 

    # we need to return the original hidden state of the policyNet for the replay memory
    pHidden = (policyNet.prevHidden[0].clone(), policyNet.prevHidden[1].clone())

    return action, pHidden, epsTh 

def nStep(exp, action, epoch, epi, ti):
    
    # global states,tHiddens,pHiddens 
    global combinaciones
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

    N_STEP = exp['step']
    GAMMA  = cfg.GAMMA
    
    # G is the inmediate discounted n_step reward: 
    # Rt+1 + gamma*Rt+2 + gamma^2*Rt+3 ...
    G = 0
    t = 0
    actionsTaken = {}
    
    # the first action is the epsilon greedy action
    actionsTaken[env.episode.currentFrame] = action
   
    sec_actions = []
    sec_actions.append(action)
    observation, reward ,terminated, truncated, infoDict = env.step(action)
    
    instant_reward = reward
    t += 1
    if cfg.CLIP_REWARD:
        G += instant_reward/500
    else:
        G += instant_reward

    updateDelay = 0
    # print('Rt: ',infoDict['timeReward'])
    # print('delay: ', env.episode.delay)
    # print('****NEXT****: ',observation)
    updateState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
    # print('up: ',updateState)
    
    # save several variables if the condition cfg.STEP_BY_STEP is True, saving the exact next time instant
    if cfg.STEP_BY_STEP:
        updateObjectsInTable = copy.deepcopy(env.oit)
        updateCurrentFrame = env.episode.currentFrame
        updateTHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
        updateDelay = env.episode.delay
        updateTargetStateDict = targetNet.state_dict()
        updateBehaviourStateDict = policyNet.state_dict()
        
    # the rest of the N-step actions, have to be optimal
    if not terminated or truncated:
        with torch.no_grad():
            for n in range(1,N_STEP):
    
                # 1 - check if env has ended
                if terminated or truncated:
                    if cfg.BORDE:
                        env.episode.terminated = False 
                        env.episode.truncated = False
                        terminated = False
                        truncated = False
                    break 
                # 2 - find the next optimal action (also advance targetNet)
                nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)
                if cfg.NO_IMPOSSIBLE_ACTIONS:
                    objects_in_table = copy.deepcopy(env.oit)
                    manipulable_objects = list(cfg.ROBOT_ACTIONS_2_OBJECTS.values())[:-1]
                    mani_obj_table = [objects_in_table[obj] for obj in manipulable_objects]
                    mani_obj_table.append(0) # la ultima accion es no hacer nada y siempre se puede hacer
                    index_posible_actions = [i for i, x in enumerate(mani_obj_table) if x == 0]
                    out = policyNet(nextState)[0]
                    nextAction = postProcessedPossibleActions(out,index_posible_actions)
                else:
                    nextAction = policyNet(nextState)[0].max(1)[1].view(1,1)
                targetNet(nextState)
                actionsTaken[env.episode.currentFrame] = nextAction.item()
                # 3 - take the a new step
                sec_actions.append(nextAction.item())
                observation, instant_reward, terminated, truncated, infoDict = env.step(nextAction.item())
                t += 1
                # 4 - update the discounted reward
                if cfg.CLIP_REWARD:
                    instant_reward = instant_reward/500
                G += (GAMMA**n) * instant_reward

    # check if the last n_step ended on a terminal state
    if terminated:
        nextState = None
        # pdb.set_trace()
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

    reward = torch.tensor([G], device=cfg.DEVICE)
    done = terminated or truncated
    
    if cfg.STEP_BY_STEP:
        st.add(epoch, epi, ti+1, 'actions_taken', actionsTaken) ######################################
    else:
        st.add(epoch, epi, ti+t, 'actions_taken', actionsTaken)
        updateTHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
    # we need the corresponding hidden of the targetLSTM to be store along nextState
    if exp['step']==1:
        updateTHidden = (targetNet.hidden[0].clone(), targetNet.hidden[1].clone())
        updateState = nextState
    
    # other options...
    # finally, update the targetNet weights
    if cfg.STEP_BY_STEP and exp['step']!=1:
        targetNet.load_state_dict(updateTargetStateDict)
        policyNet.load_state_dict(updateBehaviourStateDict)
       
        if not terminated or truncated:
            env.updateCurrentFrame(updateCurrentFrame)
            env.updateOit(updateObjectsInTable)
            env.episode.advanceTo(updateCurrentFrame)
    # print(actionsTaken)
    # print(env.oit)
    if cfg.N_STEP == 3:
        if len(sec_actions) == cfg.N_STEP:
            if combinaciones[tuple(sec_actions)]==None:
                # pdb.set_trace()
                combinaciones[tuple(sec_actions)] = [G]
            else:
                combinaciones[tuple(sec_actions)].append(G)
    # print(combinaciones)
    # pdb.set_trace()
 
    return (nextState, updateTHidden, reward, done, t, infoDict['timeReward'], infoDict['energyReward'], instant_reward, updateState, updateDelay,sec_actions)


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

    # 1 - Do nothing if the memory still don't have BATCH_SIZE samples####### no es asi
    if( len(rm) < cfg.BATCH_SIZE ):
        return 0, 0
    # if len(rm) == cfg.MEMORY_SIZE:
    #     pdb.set_trace()
    # # print(len(rm))
    # 2 - Sample a batch of n-step transitions
    if cfg.PRIORITAZED_MEMORY:
        transitions, index = rm.sample(cfg.BATCH_SIZE)
    else:
        transitions = rm.sample(cfg.BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # 3 - We need to create batches as input for the models. The problem is 
    # that when creating the nextState batch there might be None's. Thus, 
    # we need some mask to avoid passing those to the target network
    nonFinalMask = torch.tensor( 
        tuple(map(lambda s: s is not None, batch.next_state)), 
        device=cfg.DEVICE, dtype=torch.bool)
    try:
        nextStateBatch = torch.cat([s for s in batch.next_state if s is not None])   
    except:
        nextStateBatch = None
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
    if nextStateBatch != None:
        with torch.no_grad():
            tma = targetNet(nextStateBatch.unsqueeze(1), (th0, tc0))[0].squeeze(1)
            nextStateActionValues[nonFinalMask] = tma.max(1)[0]

    # 6 - Now we can compute the targets for the loss function. Here we have to
    # take care of the n-step discounts only for the model prediction (the instant rewards
    # are done in the n_step function)
        expectedStateActionValues = ( (cfg.GAMMA**exp['step']) * nextStateActionValues) + rewardBatch
    else:
        expectedStateActionValues = rewardBatch
   
    # 7 - Finally, let's compute the loss and update the model
    loss = criterion(stateActionValues, expectedStateActionValues.unsqueeze(1))

    if cfg.PRIORITAZED_MEMORY:
        # PRIORITY = ABS(LOSS)
        td_error = abs(loss.squeeze(1).detach().cpu().numpy())
        rm.update_priorities(index,td_error)
        loss = torch.mean(loss)
        # pdb.set_trace()
        
        # PRIORITY = INSPO PAPER "LEVERAGIING DEMOSNTARTYIONS FOR DEEP RL ON ROBOTICS PROBLEMS WITH SPARSE REWARDS"
        # priorities  = loss.squeeze(1).detach().cpu().numpy()**2 + cfg.EPSILON
        # rm.update_priorities(index,priorities)
        # # Calcular pk de antemano
        # pk = calculatePkValues(priorities, cfg.ALPHA)
        # # Calcular P y w para cada pi_value
        # w_values = calculatePandW(priorities, pk, cfg.ALPHA, cfg.BETA, len(priorities))
        # loss = torch.mean(loss * torch.tensor(w_values, dtype=torch.float32, device=cfg.DEVICE))


    else:
        loss = torch.mean(loss)
       
        
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
    actionsTaken = {}

    # take the action
    actionsTaken[env.episode.currentFrame] = action
    observation, reward, terminated, truncated, infoDict= env.step(action)

    # exp['stepsDone'] += 1 ##
    t = 1
    
    if cfg.CLIP_REWARD:
        reward = reward/500
    # check if the last n_step ended on a terminal state
    if terminated:
        nextState = None
    else:
        nextState = torch.tensor(observation, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

    done = terminated or truncated
    st.add(epoch, epi, ti+t, 'actions_taken', actionsTaken)

    return (nextState, reward, done, t, infoDict['timeReward'], infoDict['energyReward'])

def train(epoch, exp):
    
    # global save_path,states,tHiddens,pHiddens 
    global save_path, combinaciones
    print('Training epoch {%d/%d}' %(epoch+1, exp['cfg'].EPOCHS))

    env = exp['env']
    st  = exp['statTracker']
    cfg = exp['cfg']
    rm  = exp['replayMemory']
    policyNet = exp['policyNet']
    targetNet = exp['targetNet']
    customRandom = exp['customRandom']
    exp['stepsDone'] += 1
   
    for epi in range(env.numEpisodes):

        # 0 - reset the initial hidden states of the models
        policyNet.resetHidden()
        targetNet.resetHidden()

        # 1 - Get initial state
        if epoch != 0 and epoch % round(cfg.EPOCHS*0.1) == 0:
            env.episode.visualizeAndSave(save_path+'/graphs',epoch,mode='frames')
        state, _ = env.reset()
        
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

        # for each time step until end of episodeoptimizeModel
        t = 0
        
        while True:

            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp, customRandom)
            
            # 3.2 this is the section in charge of generating entries for the replayMemory
            # Here is where you can implement n-step strategies
            
            (nextState, tHidden, reward, done, stepsTaken, Rt, Re, instant_reward,nextInstant, updateDelay,sec_actions) = nStep(exp, action.item(), epoch, epi, t)
            # t += stepsTaken #######################################
            
            if cfg.STEP_BY_STEP:
                t += 1
            else:
                t += stepsTaken
            exp['updateTarget'] += 1
            # 3.3 store this new transition in memory

            if len(sec_actions) <  cfg.N_STEP:
                rm.push(pHidden, state, action, tHidden, None, reward)
            #     # pdb.set_trace()
            else:
                rm.push(pHidden, state, action, tHidden, nextState, reward)
           
            # 3.4 move to the next state
            # state = nextState  #######################################
            if cfg.STEP_BY_STEP:
                state = nextInstant 
            else:
                state = nextState

            # 3.5 perform policyNet learning (model optimization)
            loss,avgMaxQ = optimizeModel(exp)

            # print(exp['updateTarget'])
            # 3.6 update the targetNet
            if exp['updateTarget'] == (cfg.UPDATE_TARGET):
                updateTargetNet(exp)
                exp['updateTarget'] = 0

            # 3.7 track some useful stats
            st.add(epoch, epi,t,'epsilon', epsTh)
            st.add(epoch, epi,t,'batch_loss', loss)
            st.add(epoch, epi,t,'batch_avgMaxQ', avgMaxQ)
            st.add(epoch, epi,t, 'G', reward.item())
            st.add(epoch, epi,t,'instant_reward', instant_reward)
            st.add(epoch, epi,t, 'energy_reward', Re)
            st.add(epoch, epi,t,'time_reward',Rt)
            st.add(epoch, epi,t,'episode_durations', t)
            st.add(epoch, epi,t,'episode_delay', env.episode.delay)
            st.add(epoch, epi,t,'episode_name', env.episode.folderName)
            st.add(epoch, epi,t,'reactive_delay', env.episode.reactiveDelay)
            st.add(epoch, epi,t,'min_delay', env.episode.minDelay)
            st.add(epoch, epi, t, 'interaction_time', env.episode.interactionTime)
            st.add( epoch, epi, t, 'min_time', env.episode.minTime)
            st.add(epoch, epi,t,'reactive_time', env.episode.reactiveTime)
            st.add(epoch, epi,t, 'CA_intime', env.CA_intime)
            st.add(epoch, epi,t, 'CA_late', env.CA_late)
            st.add(epoch, epi,t, 'IA_intime', env.IA_intime)
            st.add(epoch, epi,t, 'IA_late', env.IA_late)
            st.add(epoch, epi,t, 'CI', env.CI)
            st.add(epoch, epi,t, 'II', env.II)
            # print('delay pre: ',env.episode.delay)
            # print('delay: ',updateDelay)
            
            # pdb.set_trace()
            if cfg.STEP_BY_STEP:
                if exp['step']!=1:
                    env.updateDelay(updateDelay)
            # pdb.set_trace()
            # 3.8 check if the episode is finished
            if done:    
                env.episode.storeActionsTaken(epoch, epi, st)
                
                # print("CA_intime: ",env.CA_intime)
                # print("CA_late: ",env.CA_late)
                # print("IA_intime: ",env.IA_intime)
                # print("IA_intime: ",env.IA_late)
                # print("CI: ",env.CI)
                # print("II: ",env.II)
                # pdb.set_trace()

                break
        
        # the episode is finished, let's plot stuff
        # plotEpisodeStats(st, epoch, epi)

    # we can also plot stats for this epoch 
    #plotEpochStats(st, epoch)
    # if len(rm) < cfg.MEMORY_SIZE:
    
    #         return
    if epoch % round(cfg.EPOCHS*0.01) == 0:
        
        if epoch == 0:
            FOLD = exp['fold'].split('_')[0]
            
            dt_string = exp['date'].strftime("%d-%m-%Y_%H:%M")
            
            if cfg.CLIP_REWARD:
                clip = '_clip_reward_'
            else:
                clip = ''
                
            if cfg.TEMPORAL_CONTEXT:
                temp = '_tmp_ctx_'
            else:
                temp = ''
            if cfg.PRIORITAZED_MEMORY:
                prio = '_prio_mem_'
            else:
                prio = ''
            if cfg.NO_IMPOSSIBLE_ACTIONS:
                imp = '_no_imp_actions_'
            else:
                imp =''
            if cfg.STEP_BY_STEP:
                mode = '_step_by_step'
            else:
                mode = ''
            if cfg.BORDE:
                borde = '_metiendo_borde_'
            else:
                borde = ''
            if cfg.DATASET_NAME == 'dataset_less_videos':
                dataset = 'dataset_less_videos'
            else:
                dataset = ''
           
            save_path = os.path.join('/home/sabal/N_STEP_DQN_COMPANION-main/Checkpoints/', FOLD+dataset+'_DQN_update_rate_'+str(cfg.UPDATE_TARGET) +borde+str(exp['step'])+'_STEPS_' + dt_string+temp+mode+'_SEED_'+str(cfg.SEED)+'_MEM_SIZE_'+str(cfg.MEMORY_SIZE)+prio+"NO_ACTION_PROB_"+str(cfg.NO_ACTION_PROBABILITY)+clip+imp+'_BETA_'+ str(cfg.ROBOT_TIME_BETA) + '_LAMBDA_' + str(cfg.FACTOR_ENERGY_PENALTY)
            )
            if not os.path.exists(save_path): os.makedirs(save_path)
        model_name = 'model_' + str(epoch) + '.pt'
        print("Saving model at ", os.path.join(save_path, model_name))
        
        torch.save({
        'model_state_dict': exp['policyNet'].state_dict(),
        'optimizer_state_dict': exp['optimizer'].state_dict(),
        'epoch': epoch,
        'loss': loss,
        'steps': exp['stepsDone']            
        }, os.path.join(save_path, model_name))
                
    return 

def test(epoch, exp, env, st):

    print('\tTest epoch {%d/%d}' %(epoch+1, exp['cfg'].EPOCHS))

    customRandom = exp['customRandom']
    cfg = exp['cfg']
    # env = exp['env']
    if cfg.ONLY_TEST == False:
        policyNet = exp['policyNet']
    else:
        file = os.getcwd()+'/Checkpoints/'+cfg.EXPERIMENT_NAME + "/model_"+str(epoch)+".pt"
        
        checkpoint = torch.load(file)
        policyNet = exp['policyNet']
        policyNet.load_state_dict(checkpoint['model_state_dict'])
        policyNet.eval()
    # print(exp)
    # pdb.set_trace()
    for epi in range(env.numEpisodes):
        
        # print('num epi: ',env.numEpisodes)
        # 0 - reset the initial hidden states of the models
        policyNet.resetHidden()

        # 1 - Get initial state
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=cfg.DEVICE).unsqueeze(0)

        # for each time step until end of episode
        t = 0
        while True:

            # 3.1 select an action based on current state and policy
            action, pHidden, epsTh = selectAction(state, exp, customRandom,test=True)

            # 3.2 take the step
            (nextState, reward, done, _, Rt, Re) = stepTest(exp, action.item(), epoch, epi, t, env, st)
            # reward = reward/900
            t += 1

            # 3.4 move to the next state
            state = nextState

            # 3.7 track some useful stats
            st.add(epoch, epi,t,'epsilon', epsTh)
            st.add(epoch, epi,t,'batch_loss', 0)
            st.add(epoch, epi,t,'batch_avgMaxQ', 0)
            st.add(epoch, epi,t,'instant_reward', reward)
            st.add (epoch, epi, t, 'time_reward', Rt)
            st.add(epoch,epi,t,'energy_reward', Re)
            st.add(epoch, epi,t,'episode_durations', t)
            st.add(epoch, epi,t,'episode_delay', env.episode.delay)
            st.add(epoch, epi,t,'episode_name', env.episode.folderName)
            st.add(epoch, epi,t,'reactive_delay', env.episode.reactiveDelay)
            st.add(epoch, epi,t,'min_delay', env.episode.minDelay)
            st.add(epoch, epi, t, 'interaction_time', env.episode.interactionTime)
            st.add( epoch, epi, t, 'min_time', env.episode.minTime)
            st.add(epoch, epi,t,'reactive_time', env.episode.reactiveTime)
            st.add(epoch, epi,t, 'CA_intime', env.CA_intime)
            st.add(epoch, epi,t, 'CA_late', env.CA_late)
            st.add(epoch, epi,t, 'IA_intime', env.IA_intime)
            st.add(epoch, epi,t, 'IA_late', env.IA_late)
            st.add(epoch, epi,t, 'CI', env.CI)
            st.add(epoch, epi,t, 'II', env.II)
            
            # 3.8 check if the episode is finished
            if done:    
                #print("Episode %d, delay %d" %(epi, env.episode.delay))
                
                env.episode.storeActionsTaken(epoch, epi, st)
                break

    return

def loadConfiguration():
    
    print("Loading config...")
    import config2 as cfg
    
    print("\tDone!")
    return cfg

def loadConfigurationCsv(cfg, df):
    
    cfg.FOLDS = df.FOLDS.item()
    cfg.HIDDEN_SIZE_LSTM = df.HIDDEN_SIZE_LSTM.item()
    cfg.LR = df.LR.item()
    cfg.MEMORY_SIZE = df.MEMORY_SIZE.item()
    cfg.PRIORITAZED_MEMORY = df.PRIORITAZED_MEMORY.item()
    cfg.ROBOT_PROB_FAILURE = df.ROBOT_PROB_FAILURE.item()
    cfg.ROBOT_TIME_BETA = df.ROBOT_TIME_BETA.item()
    cfg.FACTOR_ENERGY_PENALTY = df.FACTOR_ENERGY_PENALTY.item()
    cfg.IMPOSSIBLE_ACTION_PENALTY = df.IMPOSSIBLE_ACTION_PENALTY.item()
    cfg.N_STEP = df.N_STEP.item()
    cfg.GAMMA = df.GAMMA.item()
    cfg.EPS_START = df.EPS_START.item()
    cfg.EPS_END = df.EPS_END.item()
    cfg.EPS_DECAY = df.EPS_DECAY.item()
    cfg.TAU = df.TAU.item()
    try:
        cfg.TEMPORAL_CONTEXT = df.TEMPORAL_CONTEXT.item()
    except:
        print('No temporal context content in cfg file')
        pass
    try:
        df.CLIP_REWARD = df.CLIP_REWARD.item()
    except:
        print('No clip reward content in cfg file')
        pass
    
    return cfg
def initExperiment(cfg, fold, step):
    
    global save_path
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
    # torch.manual_seed(10)
    policyNet = DQN_LSTM_LateFusion(state_dimensionality, cfg.HIDDEN_SIZE_LSTM, num_actions).to(cfg.DEVICE)
    # torch.manual_seed(10)
    targetNet = DQN_LSTM_LateFusion(state_dimensionality, cfg.HIDDEN_SIZE_LSTM, num_actions).to(cfg.DEVICE)

    #policyNet = DQN_MLP(state_dimensionality,  num_actions).to(cfg.DEVICE)
    #targetNet = DQN_MLP(state_dimensionality, num_actions).to(cfg.DEVICE)

    targetNet.load_state_dict(policyNet.state_dict())
    optimizer = optim.AdamW(policyNet.parameters(), lr = cfg.LR,amsgrad=True) 
    
    # 3 - Initialize stat tracking objects
    st = StatTracker()
    stVal = StatTracker()
    stTest = StatTracker()

    # 4 - Initialize replay memory
    if cfg.PRIORITAZED_MEMORY:
        rm = PrioritizedReplayMemory(cfg.MEMORY_SIZE)
        criterion = nn.SmoothL1Loss(reduction='none')
        
    else:
        rm = ReplayMemory(cfg.MEMORY_SIZE)
        criterion = nn.SmoothL1Loss()
        
    full = rangeRandom()
    now = datetime.now()
    updateTarget = 0
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
           'replayMemory': rm,
           'date':now,
           'step':step,
           'updateTarget': updateTarget,
           'customRandom': full
           
    }
           
    

    print("\tDone!")
    return exp

def runExperiment(cfg):

    global save_path, combinaciones

    # Crear un diccionario para almacenar las combinaciones
    
    # print(combinaciones)
    # pdb.set_trace()
    for i,fold in enumerate(cfg.FOLDS):
        print("Starting experiment: %s" %(fold))
        combinaciones = {}
        
        if cfg.N_STEP == 3:
            # Rango de números del 0 al 5
            numeros = range(6)
            
            # Generar todas las combinaciones posibles de tres elementos
            for num1 in numeros:
                for num2 in numeros:
                    for num3 in numeros:
                        # Crear una clave única para cada combinación
                        clave = (num1, num2, num3)
                        
                        # Agregar la combinación al diccionario
                        combinaciones[clave] = None  # Puedes asignar un valor específico si es necesario
        # 1 - Init experiment env, models, opt, loss, stat objects, etc
        exp = {}
        exp_i = initExperiment(cfg, fold, cfg.N_STEP)
        # exp_i['fold'] = fold
        env = exp_i['env']
        #TENGO QUE GURADAR CUANDO ENTRENO LA OCNFIGURACION DEL EXPERIMENTO, TODO EL EXP, Y ASI LUEGO LEERLO AQUI
        for e in range(cfg.EPOCHS):
            train(e, exp_i)
            if e == 0:
                if not os.path.exists(save_path+'/graphs'): os.makedirs(save_path+'/graphs')
            if cfg.N_STEP == 3:
                if e%100==0:
                    # Crear un diccionario para almacenar el conteo de elementos por clave
                    conteo_por_clave = {}
                    # Recorrer las claves y contar la cantidad de elementos en cada lista
                    for clave, valor in combinaciones.items():
                        if isinstance(valor, list):  # Verificar si el valor es una lista
                            cantidad_elementos = len(valor)
                            conteo_por_clave[clave] = cantidad_elementos
                        # else:
                        #     conteo_por_clave[clave] = 0  # Otra opción, si el valor no es una lista, asignar 0
                    
                    # Imprimir el conteo de elementos de mayor a menor
                    conteo_ordenado = sorted(conteo_por_clave.items(), key=lambda x: x[1], reverse=True)
                    
                    for clave, cantidad_elementos in conteo_ordenado:
                        print(f"Clave: {clave}, Cantidad de Elementos: {cantidad_elementos}")
           
            # test the model in both train/test partitons (no exploration)
            test(e, exp_i, exp_i['envVal'], exp_i['statTrackerVal'])
            
            # test(e, exp_i, exp_i['envTest'], exp_i['statTrackerTest'])
            # print(exp_i['statTrackerTest'].get('instant_reward'))
            # pdb.set_trace()

        save_reward = exp_i['statTrackerVal'].get('instant_reward')
        save_enery_reward = exp_i['statTrackerVal'].get('energy_reward')
        save_time_reward = exp_i['statTrackerVal'].get('time_reward')
        save_time_interaction = exp_i['statTrackerVal'].get('episode_durations')
        save_delay = exp_i['statTrackerVal'].get('episode_delay')
        save_episode_name = exp_i['statTrackerVal'].get('episode_name')
        save_reactive_delay = exp_i['statTrackerVal'].get('reactive_delay')
        save_min_delay = exp_i['statTrackerVal'].get('min_delay')
        save_reactive_time = exp_i['statTrackerVal'].get('reactive_time')
        save_min_time = exp_i['statTrackerVal'].get('min_time')
        save_interaction_time = exp_i['statTrackerVal'].get('interaction_time')
        save_CA_intime =  exp_i['statTrackerVal'].get('CA_intime')
        save_CA_late =  exp_i['statTrackerVal'].get('CA_late')
        save_IA_intime =  exp_i['statTrackerVal'].get('IA_intime')
        save_IA_late =  exp_i['statTrackerVal'].get('IA_late')
        save_CI =  exp_i['statTrackerVal'].get('CI')
        save_II =  exp_i['statTrackerVal'].get('II')
       
        
        data_test = {
        'save_time_interaction': save_time_interaction,
        'energy_reward': save_enery_reward,
        'time_reward':save_time_reward,
        'reward':save_reward,
        'delay':save_delay,
        'minimum_delay': save_min_delay,
        'reactive_delay': save_reactive_delay,
        'episode_name': save_episode_name,
        'reactive_time': save_reactive_time,
        'minimum_time': save_min_time,
        'interaction_time': save_interaction_time,
        "CA_intime": save_CA_intime,
        "CA_late": save_CA_late,
        "IA_intime":save_IA_intime,
        "IA_late" :save_IA_late,
        "CI" :save_CI,
        "II":save_II 
        }

        df_test = pd.DataFrame(data_test)
        df_test.to_csv(save_path+'/results_val_raw.csv')
                
        
        data_cfg = {
                    'FOLDS': [exp_i['fold']],
                    'HIDDEN_SIZE_LSTM': [exp_i['cfg'].HIDDEN_SIZE_LSTM],
                    'LR' :[exp_i['cfg'].LR],
                    'EPOCHS':[exp_i['cfg'].EPOCHS],
                    'NUM_EPISODES' :[exp_i['cfg'].NUM_EPISODES],
                    'BATCH_SIZE' :[exp_i['cfg'].BATCH_SIZE],
                    'GAMMA' :[exp_i['cfg'].GAMMA],
                    'EPS_START' :[exp_i['cfg'].EPS_START],
                    'EPS_END' :[exp_i['cfg'].EPS_END],
                    'EPS_DECAY' :[exp_i['cfg'].EPS_DECAY],
                    'TAU' :[exp_i['cfg'].TAU],
                    'N_STEP' :[exp_i['cfg'].N_STEP],
                    'MEMORY_SIZE' :[exp_i['cfg'].MEMORY_SIZE],
                    'PRIORITAZED_MEMORY' :[exp_i['cfg'].PRIORITAZED_MEMORY],
                    'ROBOT_TIME_BETA' :[exp_i['cfg'].ROBOT_TIME_BETA],
                    'ROBOT_PROB_FAILURE' :[exp_i['cfg'].ROBOT_PROB_FAILURE],
                    'FACTOR_ENERGY_PENALTY' :[exp_i['cfg'].FACTOR_ENERGY_PENALTY],
                    'IMPOSSIBLE_ACTION_PENALTY' :[exp_i['cfg'].IMPOSSIBLE_ACTION_PENALTY],
                    'NO_ACTION_PROBABILITY':[exp_i['cfg'].NO_ACTION_PROBABILITY],
                    'TEMPORAL_CONTEXT': [exp_i['cfg'].TEMPORAL_CONTEXT],
                    'CLIP_REWARD': [exp_i['cfg'].CLIP_REWARD]
                    }
        # print(data_cfg)
        # pdb.set_trace()
        df_cfg = pd.DataFrame(data_cfg)
        df_cfg.to_csv(save_path +'/cfg.csv')
        if cfg.N_STEP == 3:
            with open(save_path+'/dict_actions_G.pkl', 'wb') as archivo_pickle:
                pickle.dump(combinaciones, archivo_pickle)
            visualizeSecActions(save_path)
        proccessCsv(save_path, '0', 'val')
        plotActions(save_path, '/results_test_proccessed_val0.csv')
        exp[i+1] = exp_i
    if cfg.TEMPORAL_CONTEXT:
        ROBOT_EXECUTION_TIMES  = env.get_robot_execution_times()
        with open(save_path+'/robot_execution_times', 'wb') as handle:
            pickle.dump(ROBOT_EXECUTION_TIMES, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\tDone!")
    return exp

def runTest(cfg):

    # global save_path
    
    exp = {}
    

    file = os.getcwd() + '/Checkpoints/' + cfg.EXPERIMENT_NAME + '/cfg.csv'
        
    if os.path.exists(file):
        df = pd.read_csv(file)
        # print(df)
            # print('n-step: ',cfg.N_STEP)
        cfg=loadConfigurationCsv(cfg,df)
    # 1 - Init experiment env, models, opt, loss, stat objects, etc
    fold = cfg.FOLDS
    exp_i = initExperiment(cfg, fold, cfg.N_STEP)
    
    # Path to the folder containing the files
    # print(os.getcwd())
    folder_path = os.getcwd() + '/Checkpoints/' + cfg.EXPERIMENT_NAME
    
    # Get the list of files in the folder
    files = os.listdir(folder_path)
    
    # Regular expression to extract numbers from the file names
    pattern = re.compile(r'model_(\d+)\.pt')
    
    # List to store the numbers from the files
    file_numbers = []
    
    print("Loading experiment: %s" % (folder_path))
    print('fold test: ', fold)
    # Iterate over the files and extract the numbers
    for file in files:
        match = pattern.match(file)
        if match:
            number = int(match.group(1))
            file_numbers.append(number)
    
    file_numbers_sorted = sorted(file_numbers)

    # pdb.set_trace()
    print('N STEP: ', cfg.N_STEP)
    for e in file_numbers_sorted:
        
        test(e, exp_i, exp_i['envTest'], exp_i['statTrackerTest'])
        
        
    # save_G = exp_i['statTrackerTest'].get('G')
    save_reward = exp_i['statTrackerTest'].get('instant_reward')
    save_enery_reward = exp_i['statTrackerTest'].get('energy_reward')
    save_time_reward = exp_i['statTrackerTest'].get('time_reward')
    save_time_interaction = exp_i['statTrackerTest'].get('episode_durations')
    save_delay = exp_i['statTrackerTest'].get('episode_delay')
    save_episode_name = exp_i['statTrackerTest'].get('episode_name')
    save_reactive_delay = exp_i['statTrackerTest'].get('reactive_delay')
    save_min_delay = exp_i['statTrackerTest'].get('min_delay')
    save_reactive_time = exp_i['statTrackerTest'].get('reactive_time')
    save_min_time = exp_i['statTrackerTest'].get('min_time')
    save_interaction_time = exp_i['statTrackerTest'].get('interaction_time')
   
    save_CA_intime =  exp_i['statTrackerTest'].get('CA_intime')
    save_CA_late =  exp_i['statTrackerTest'].get('CA_late')
    save_IA_intime =  exp_i['statTrackerTest'].get('IA_intime')
    save_IA_late =  exp_i['statTrackerTest'].get('IA_late')
    save_CI =  exp_i['statTrackerTest'].get('CI')
    save_II =  exp_i['statTrackerTest'].get('II')
    
    data_test = {
    
    'save_time_interaction': save_time_interaction,
    'energy_reward': save_enery_reward,
    'time_reward':save_time_reward,
    'reward':save_reward,
    'delay':save_delay,
    'minimum_delay': save_min_delay,
    'reactive_delay': save_reactive_delay,
    'episode_name': save_episode_name,
    'reactive_time': save_reactive_time,
    'minimum_time': save_min_time,
    'interaction_time': save_interaction_time,
    # 'G': save_G
    "CA_intime": save_CA_intime,
    "CA_late": save_CA_late,
    "IA_intime":save_IA_intime,
    "IA_late" :save_IA_late,
    "CI" :save_CI,
    "II":save_II 
    
    }

    df_test = pd.DataFrame(data_test)
    df_test.to_csv(folder_path+'/results_test_raw_'+str(cfg.NUM_LOOP)+'.csv')
    
    proccessCsv(folder_path, cfg.NUM_LOOP, 'test')
    
    plotActions(folder_path, '/results_test_proccessed_0.csv')
    
            
    print("\tDone!")
    return exp
def saveAndVisualizePerEpoch(cfg, exp):

    print("Saving per epoch results")

    epochs = range(cfg.EPOCHS)
    rewardVal = []
    rewardTest = []
    delayVal = []
    delayTest = []

    for epoch in epochs:

        rewardVal_e, rewardTest_e, _ = _recoverKFoldStat(cfg, exp, 'instant_reward', epoch=epoch, func='sum', reduction=True)
        delayVal_e, delayTest_e, _ = _recoverKFoldStat(cfg, exp, 'episode_delay', epoch=epoch, func='last', reduction=True)

        rewardVal.append(np.mean(rewardVal_e))
        rewardTest.append(np.mean(rewardTest_e))
        delayVal.append(np.mean(delayVal_e))
        delayTest.append(np.mean(delayTest_e))

    p = 'figs/epochs/'
    plotHelper3( epochs, rewardVal, epochs, rewardTest, 'epoch', 'average reward', 'Average reward per epoch', ['train', 'test'], p+'rewards.png')
    plotHelper3( epochs, delayVal, epochs, delayTest, 'epoch', 'average delay', 'Average delay per epoch', ['train', 'test'], p+'delays.png')
    return

def saveAndVisualizePerRecipe(cfg, exp, epoch=0):
    
    print("Saving per recipe results")
    
    # 1 - Get all recipe names alfabetically sorted
    rewardVal, rewardTest, recipeNames = _recoverKFoldStat(cfg, exp, 'instant_reward', epoch=epoch, func='sum', reduction=True)
    aTakenVal, aTakenTest, _ = _recoverKFoldStat(cfg, exp, 'actions_taken', epoch=epoch, func='notIdleActions', reduction=True)
    delayVal, delayTest, _ = _recoverKFoldStat(cfg, exp, 'episode_delay', epoch=epoch, func='last', reduction=True)
    reactiveTimes, _, _ = _recoverKFoldStat(cfg, exp, 'reactive_time', epoch=epoch, func='last', reduction=True)
    minDelays, _, _ = _recoverKFoldStat(cfg, exp, 'min_delay', epoch=epoch, func='last', reduction=True)

    plotHelper2(rewardVal, rewardTest, '#recipe', 'reward', 'total reward per recipe', 
                ['val', 'test'], recipeNames, 'figs/allFolds/total_reward_epoch_%d.png' %epoch)
    plotHelper2(aTakenVal, aTakenTest, '#recipe', '#actions', 'total actions taken (!=5) per recipe', 
                ['val', 'test'], recipeNames, 'figs/allFolds/actions_taken_epoch_%d.png' %epoch)
    plotHelper2(delayVal, delayTest, '#recipe', 'delay (frames)', 'total delay per per recipe', 
                ['val', 'test', 'reacTime', 'oracle'], recipeNames, 'figs/allFolds/delay_epoch_%d.png' %epoch, reactiveTimes, minDelays)
    
    # save all the recipes (test) to visualize the kind of actions the robot took
    for k in exp.keys():
        env = exp[k]['envTest']
        for i in range(env.numEpisodes):
            epi = next(env.allEpisodes)
            epi.visualizeAndSave(mode='frames')

    print("\tDone!")
    return

def _recoverKFoldStat(cfg, exp, statName, epoch=0, func='sum', reduction=True):

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
        valCols  = [r2c[r[0]] for r in stVal.get('episode_name', epoch)]
        testCols = [r2c[r[0]] for r in stTest.get('episode_name', epoch)]
        valRows  = [i] * len(valCols)

        valuesVal  = [_applyFunc(epi, func) for epi in stVal.get(statName, epoch)]
        valuesTest = [_applyFunc(epi, func) for epi in stTest.get(statName, epoch)]

        # lets fill the valMat and testMat
        valMat[valRows, valCols] = valuesVal
        testMat[testCols] = valuesTest

    if reduction is not None and valMat is not None:
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
    

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.cuda.manual_seed_all(cfg.SEED)

    # # Configurar la reproducibilidad de operaciones determinísticas si usas CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # torch.nn.init.manual_seed(10)
    # 2 - Init experiment env, models, opt, loss, stat objects, etc.
    #exp = initExperiment(cfg)

    # 3 - Run the training/validation process
    if cfg.ONLY_TEST == False:
        exp = runExperiment(cfg)
    else:
        file = os.getcwd() + '/Checkpoints/' + cfg.EXPERIMENT_NAME + '/cfg.csv'
        
        if os.path.exists(file):
            df = pd.read_csv(file)
            # print(df)
            # print('n-step: ',cfg.N_STEP)
            cfg=loadConfigurationCsv(cfg,df)
            # print('n-step: ',cfg.N_STEP)
            # pdb.set_trace()
        
        exp = runTest(cfg)
    

    # 4 - Save and visualize results
    #saveAndVisualizePerRecipe(cfg, exp, epoch=cfg.EPOCHS-1)
    saveAndVisualizePerEpoch(cfg, exp)