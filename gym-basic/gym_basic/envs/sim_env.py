import copy
import os
import sys
import glob
import random

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from itertools import cycle, count

import config2 as cfg
from utils import readTestTxt

class AtomicAction():

    def __init__(self, index, actionLabel, objectsLabels, frameInit, frameEnd, requiresRobot, requiredObj):

        self.index = index
        self.actionLabel = actionLabel
        self.objectsLabels = objectsLabels
        self.frameInit = frameInit
        self.frameEnd = frameEnd
        self.duration = frameEnd - frameInit
        self.requiresRobot = requiresRobot
        self.requiredObj  = requiredObj
        return
    
    def __str__(self):
    
        print("atomic action number = %d" %self.index)
        print("Action = %s" %(cfg.ATOMIC_ACTIONS_MEANINGS[self.actionLabel]))
        print("Objects involved")
        for object in self.objectsLabels:
            print("\t%s" %(cfg.OBJECTS_MEANINGS[object[0]]))

        print("frame init = %d" %(self.frameInit))
        print("frame end = %d" %(self.frameEnd))
        print("requires robot = %s" %(str(self.requiresRobot)))  
        print("required object = %s" %(self.requiredObj)) 

        return ""
    
class Episode():

    def __init__(self, folderPath, labelsPath, actions2Remove, robotNeededActions,
                 videoFPS, featFPS, selFeat):
    
        
        self.folderPath = folderPath
        self.folderName = folderPath[folderPath.rfind('/')+1::]
        self.labelsPath = labelsPath
        self.actions2Remove = actions2Remove
        self.robotNeededActions = robotNeededActions
        self.videoFPS = videoFPS
        self.featFPS = featFPS
        self.selFeat = range(selFeat)
        self.atomicActions = []
        self.frameFeatures = None 
        self.lastFrame = None
        self.reactiveTime = 0
        self.minDelay = 0

        # attributes that must be reset each time the episode starts
        self.currentFrame = 0
        self.delay = 0
        self.terminated = True
        self.truncated = True
        self.actionsTaken = {}

        # load the episode info
        self.loadEpisode()
        return
    
    def reset(self):
        self.currentFrame = 0
        self.delay = 0
        self.terminated = False
        self.truncated = False
        self.actionsTaken = {}
        return
    
    def advanceTo(self, f):

        if f > self.lastFrame:
            self.currentFrame = self.lastFrame
            self.terminated = True
        else:
            self.currentFrame = f

        return
    
    def skipFrames(self, numFrames):

        self.advanceTo(self.currentFrame + numFrames)
        return
    
    def findNextRobotAction(self):

        # this function returns the next atomic action that needs the robot
        # to do something. If there is no need for robot intervention for
        # the rest of the recipe, it returns a default atomic action

        ac = AtomicAction(0,0,0,np.inf, np.inf, False, None)

        for ac_i in self.atomicActions:
            if ac_i.frameInit >= self.currentFrame and ac_i.requiresRobot:
                return ac_i
        
        return ac
    
    def findNextAtomicAction(self, ac):

        for ac_i in self.atomicActions:
            if ac_i.index > ac.index:
                return ac_i
        
        # there is no next atomic action
        return None
    
    def isObjectNecessary(self, objName):

        #objName = cfg.OBJECTS_MEANINGS[objLabel]

        for ac_i in self.atomicActions:
            if ac_i.requiresRobot and ac_i.requiredObj == objName:
                return True
        
        return False
    
    def findNextRewardFrame(self):

        # This function returns the frame where we have the next reward time 
        # instant. All starts and ends of atomic actions are reward instants
        for ac_i in self.atomicActions:
            
            if self.currentFrame < ac_i.frameInit:
                return ac_i.frameInit
            
            if self.currentFrame < ac_i.frameEnd:
                return ac_i.frameEnd
            
        return self.lastFrame
    
    def fps2time(self, frameNum):
        return frameNum*1.0 / self.videoFPS
    
    def time2fps(self, time):
        return int(time * self.videoFPS)

    def getTotalLength(self, mode='frames'):

        assert mode=='frames' or mode == 'time'

        totalFrames = self.frameFeatures.shape[0]
        return totalFrames if mode == 'frames' else self.fps2time(totalFrames)
    
    def getOptimalLength(self, mode='frames'):

        totalFrames = self.getTotalLength(mode='frames')
        
        for ac in self.atomicActions:
            if ac.requiresRobot:
                totalFrames -= ac.duration

        return totalFrames if mode == 'frames' else self.fps2time(totalFrames)

    def loadEpisode(self):

        # 1 - Read frame features
        framePathsList = sorted(glob.glob(os.path.join(self.folderPath, 'frame_*')))
        self.frameFeatures = np.zeros( (len(framePathsList), len(self.selFeat)), dtype=np.float32 )

        for i,framePath in enumerate(framePathsList):

            fFeat = np.load(framePath, allow_pickle=True)['data']
            self.frameFeatures[i,:] = fFeat[self.selFeat]

        # 2 - Replicate the features to account for the difference between 
        # the featFPS and the videoFPS
        replicateFactor = self.videoFPS / self.featFPS
        self.frameFeatures = np.repeat(self.frameFeatures, replicateFactor, axis=0)

        # 3 - Read labels
        pandasDF = np.load(self.labelsPath, allow_pickle=True)
        #print(pandasDF)

        # 4 - Fill atomic actions
        keep = np.full(self.frameFeatures.shape[0], True, dtype=bool)
        offset, numAc = 0,0
        for index, row in pandasDF.iterrows():

            # read pandas row
            actionIndex  = row.label
            objectsIndex = row.label_obj
            frameInit    = row.frame_init
            frameEnd     = row.frame_end

            duration       = frameEnd - frameInit
            requiresRobot  = row.label in self.robotNeededActions
            actionToRemove = row.label in self.actions2Remove

            if actionToRemove:
                offset += duration
                keep[frameInit:frameEnd] = False
                continue

            requiredObj = cfg.HUMAN_ACTIONS_2_OBJECTS[actionIndex] if requiresRobot else None

            # create the atomic action and add it to the episode
            self.atomicActions.append(AtomicAction(numAc, actionIndex, objectsIndex, 
                frameInit-offset, frameEnd-offset, requiresRobot, requiredObj))
            numAc += 1

        # Finally, remove the frames that belong to self.actions2Remove
        self.frameFeatures = self.frameFeatures[keep]
        self.lastFrame = self.frameFeatures.shape[0]-1
        return

    def __str__(self):

        print("EPISODE INFO")
        print("\tFolder path %s" %self.folderPath)
        print("\tHuman actions that needs robot: ", self.robotNeededActions)
        print("\tCurrent frame in the episode %d" %(self.currentFrame) )
        print("\nAtomic actions: ")

        for ac in self.atomicActions:
            print(ac)

        return ""
    
    def storeActionsTaken(self, epoch, epi, st):

        self.actionsTaken = {k: v for t in st.data[epoch][epi].keys() 
            for k, v in st.data[epoch][epi][t]['actions_taken'].items()}
        
        return
    
    def visualizeAndSave(self, mode='time'):

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set the y-axis limit and labels
        ax.set_ylim(0.11, 0.31)
        ax.set_yticks([])

        xticks = []
        xticklabels = []

        # Plot the timeline
        for ac in self.atomicActions:

            if mode == 'time':
                duration = round(self.fps2time(ac.duration))
                startTime = round(self.fps2time(ac.frameInit))
            else:
                duration = round(ac.duration)
                startTime = round(ac.frameInit)

            name = "a" + str(ac.index)
            
            c = 'r' if ac.requiresRobot else 'b'
            f = (1.0,0.8,0.8) if ac.requiresRobot else (1.0,1.0,1.0)

            # Create a rectangle for the action
            rect = patches.Rectangle((startTime, 0.1), duration, 0.3, linewidth=1, edgecolor=c, facecolor=f)
            
            # Add the action index and action label
            ax.text(startTime + duration / 2, 0.20, name, ha='center', va='center', fontsize=10, color=c)
            ax.text(startTime + duration / 2, 0.15, ac.actionLabel, ha='center', va='center', fontsize=10, color=c)

            ax.add_patch(rect)

            # Store the x-axis tick positions and labels
            xticks.extend([startTime, startTime + duration])
            xticklabels.extend([str(startTime), str(startTime + duration)])

        for k,v in self.actionsTaken.items():
            ax.text(k, 0.30, v, ha='center', va='center', fontsize=14, color='g')

        # Set the x-axis limit
        if mode == 'time':
            ax.set_xlim(0, self.getTotalLength(mode='time'))
        else:
            ax.set_xlim(0, self.getTotalLength(mode='frames'))

        # Configure the figure size
        fig.set_size_inches(13, 1)

        # Set x-axis ticks and labels
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=0, fontsize=8)
        ax.tick_params(axis='x', which='both', bottom=False, top=False)

        for i,label in enumerate(ax.get_xticklabels()):
            if i % 2 == 0:
                label.set_y(-0.3)

        # Save the timeline as a PNG image
        ax.set_title(self.folderName)
        imName = 'figs/recipes/' + self.folderName + '.png'
        plt.savefig(imName, bbox_inches='tight', dpi=300)
        #plt.show()
        return

class SimulatedEnv(gym.Env):

    """
    This enviroment simulates a real robot serving items to a human making different 
    foods. On each episode, the human is making something to eat (toast, drink, cereals).
    There are a series of objects in a table where the human is working. This set of 
    objects changes on each episode (video). There is also a set of objects in the fridge. 
    This set of objects is always the same for all episodes. The person is wearing glasses
    with pupil tracking, and an ANN is making some predictions based on the human gaze from
    current and past time instants. 
    
    """

    def __init__(self, mode, testFile):
        super(SimulatedEnv, self).__init__()

        #print("importo cfg")
        #root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        #sys.path.append(root_path)
        #import config2 as cfg

        """
        Mandatory attributes
            action_space: the Space object corresponding to valid actions
            observation_space: the space object corresponding to valid observations
            metadata: a dict containing the rendering modes, fps, etc.
            render_mode: the render mode of the env determined at initialization
            reward_range: a tuple with the minimum and maximum possible rewards
        """

        # 1 - attributes
        self.mode = mode
        self.testfile = testFile
        self.allEpisodes = []
        self.numEpisodes = None
        self.episode = None  
        self.oit = copy.deepcopy(cfg.OBJECTS_INIT_STATE)
        
        # 2 - set action and observation spaces
        stateDims = cfg.ANNOTATIONS_DIM + len(cfg.OBJECTS_INIT_STATE)
        self.stateDims = stateDims
        low = -np.inf * np.ones(stateDims)
        high = np.inf * np.ones(stateDims)

        self.action_space = spaces.Discrete(cfg.NUM_ROBOT_ACTIONS)
        self.observation_space = spaces.Box(low = low, high = high, dtype=np.float32)

        # 3 - read episodes
        self.loadEpisodes()
        self.computeEpisodesReactiveTime()
        self.computeEpisodesMinDelay()
        return
    
    def computeEpisodesReactiveTime(self):

        """
        This function computes the times necessary for a recipe to be completed
        under a reactive robot (a robot that only acts via ASR)
        """
        for i in range(self.numEpisodes):
            
            self.reset()
            while self.episode.terminated is False:
                self.step(5)

            self.episode.reactiveTime = self.episode.delay
            #self.episode.reset()
        
        return
    
    def computeEpisodesMinDelay(self):

        """
        This function computes for each episode, the minimun delay the robot can 
        cause. Note that this might change on each iteration given that the times
        taken by the robot are stochastic. 
        """
        for i in range(self.numEpisodes):

            self.reset()
            requiredActions = []

            # find the necessary actions to be taken
            for ac_i in self.episode.atomicActions:
                if ac_i.requiresRobot:
                    requiredActions.append(cfg.ROBOT_OBJECTS_2_ACTIONS[ac_i.requiredObj])

            # now, take the actions
            while len(requiredActions) > 0:
                action = requiredActions.pop(0)
                self.step(action)

            # do nothing for the rest of the episode
            while self.episode.terminated is False:
                self.step(5)

            self.episode.minDelay = self.episode.delay
            #self.episode.reset()

        return
    
    def loadEpisodes(self):

        # read episode folders
        dbFolder = os.path.join(cfg.ANNOTATIONS_FOLDER, cfg.DATASET_NAME)
        testPath = os.path.join(cfg.ANNOTATIONS_FOLDER, self.testfile)
        
        allFolders  = glob.glob(dbFolder+"/*")
        testFolders = [os.path.join(dbFolder, testFolder) for testFolder in readTestTxt(testPath)]

        # keep train or test folders
        folders = testFolders if self.mode == 'test' else list( set(allFolders) - set(testFolders) )
        folders = sorted(folders) # sort them or shuffle them
        
        # load the episodes
        e = []
        for f in folders:

            labelsPath = os.path.join(f, cfg.LABELS_FILE_NAME)
            e.append( Episode(f, labelsPath, cfg.ACTIONS_TO_REMOVE_FROM_RECIPES, 
                cfg.ROBOT_NEEDED_FOR_ACTIONS, cfg.VIDEO_FPS, cfg.ANNOTATIONS_FPS, 
                cfg.ANNOTATIONS_DIM))

        self.numEpisodes = len(e)
        self.allEpisodes = cycle(e)
        return

    def isImposibleAction(self, action):

        """
        this function checks whether the action that the robot wants to
        take its possible or not (because the object is not in the fridge)
        Returns True if the action is impossible to execute.
        """
        objTaken = cfg.ROBOT_ACTIONS_2_OBJECTS[action]
        
        if objTaken == 'do nothing': 
            return False
        else:
            return self.oit[objTaken] == 1    

    def buildState(self, randomDelay=30):

        # the random delay is there to avoid always taking the same
        # frames, either the 0 frame or the same exact reward instants.
        # Pass 0 if you want to avoid this random delay
        delay = random.randint(0, randomDelay)

        # we need to check that the delay doesn't cause the time to advance
        # beyond an ASR event
        futureTime = self.episode.currentFrame + delay
        ac = self.episode.findNextRobotAction()
        maxTime = min(futureTime, ac.frameInit)

        self.episode.advanceTo(maxTime)
        #self.episode.skipFrames(delay)
        if cfg.ENV_VERBOSE: print("\t\tRandom delay = %d" %delay)

        feat = self.episode.frameFeatures[self.episode.currentFrame]
        oit = np.array(list(self.oit.values()), dtype=np.float32)
        return np.concatenate((feat, oit), axis=0)
    
    def reset(self, options={}):

        """
        This method is going to move to the next episode (or the first episode if this is the
        first time it is called). Returns:
            observation: the initial state (analogous to the observation returned by step)
            info:        a dict with auxiliary info complementing observation (analogous
                         to the one returned by step)
        """

        # 1 - Select and reset the next episode
        self.episode = next(self.allEpisodes)
        self.episode.reset()

        # 2 - reset objects in table
        self.oit = copy.deepcopy(cfg.OBJECTS_INIT_STATE)

        # 3 - build the next state and info
        state = self.buildState()

        if cfg.ENV_VERBOSE:
            self.episode.visualizeAndSave(mode='frames')
            print("Recipe: %s" %(self.episode.folderPath))

        return state, {}
    
    def step(self, action):

        """
        This method receives an action that the agent wants to take and 
        returns:
            observation: the next state the agent encounters for taking 'action' (vector)
            reward: the reward as a result of taking the 'action' (float)
            terminated: whether the agent has reached terminal state (bool) 
            truncated: a float indicating a time-limit reached or out of bounds agent that 
                        makes the enviroment to finish although a terminal state might no be 
                        reached
            info:     a dictionary with auxiliary diagnostic info (like individual reward terms
                        that are mixed in the final reward)
        """
        
        # we have to call reset when an episode ends
        assert self.episode.terminated == False, "The episode must be reset before calling step the first time"
        assert 0 <= action <= 5, "Action must be between 0 and 5"

        if cfg.ENV_VERBOSE:
            print("Robot taking action %d: %s" %(action, cfg.ROBOT_ACTIONS_MEANINGS[action]))

        if cfg.ROBOT_ACTIONS_MEANINGS[action] == 'do nothing':
            Rt, Re, R, nextState = self.stepIdle(action)
        else:
            Rt, Re, R, nextState = self.stepAction(action)

        # create the info dict and return the rewards and nextState
        if cfg.ENV_VERBOSE:
            print("Next state is: %d" %(self.episode.currentFrame))
            print("Rt = %.4f, Re = %.4f, R = %.4f" %(Rt, Re, R))
            print("total recipe delay (in frames): %d" %(self.episode.delay))

        
        infoDict = {'timeReward': Rt, 'energyReward': Re}

        return nextState, R, self.episode.terminated, self.episode.truncated, infoDict
    
    def stepIdle(self, action):

        # if the robot decides to stay idle, we need to check if we are going
        # to need something from the robot before the next decision time step 
        ac = self.episode.findNextRobotAction()
        frames2NeedRobot = ac.frameInit - self.episode.currentFrame

        if cfg.ENV_VERBOSE:
            print("\tAction taken in frame: %d" %(self.episode.currentFrame))
            print("\tStill %s frames to next robot action" %(str(frames2NeedRobot)))

        # for staying idle the reward is zero unless a human needs to wait
        timeReward, energyReward, reward = 0,0,0

        # in this case, there will be a human needsRobot atomic action event
        # before the robot has a new chance to make a decision
        if frames2NeedRobot <= cfg.DECISION_RATE:
            
            # if necessary object is NOT on the table
            if self.oit[ac.requiredObj] != 1:

                if cfg.ENV_VERBOSE: 
                    print("\tNecessary object not on the table: ASR")
            
                timeReward, energyReward, reward = self.computeReward(
                    ac.frameInit, ac.frameInit, ac.requiredObj, ac.requiredObj, 
                    ASR = True)
                self.oit[ac.requiredObj] = 1

                # advance to start of need robot ASR
                self.episode.advanceTo(ac.frameInit)
            
            else:
                # the necessary object was on the table, thus, advance frames normally
                self.episode.skipFrames(cfg.DECISION_RATE)

        else:

            if cfg.ENV_VERBOSE: 
                print("\tStill time til next robot action")

            self.episode.skipFrames(cfg.DECISION_RATE)
            
        # compute nextState
        nextState = self.buildState(randomDelay=0)
        return timeReward, energyReward, reward, nextState
    
    def stepAction(self, action):
        
        """
        This method is called when the robot decides to take an action different
        than staying idle. The logic of the rewards and next states is different
        depending on whether the robot action will finish before or after the 
        next time the human needs something from the robot
        """

        framesNeededRobot = self.computeRobotTime(action)
        robotActionEnds = self.episode.currentFrame + framesNeededRobot
        ac = self.episode.findNextRobotAction()

        # robot action ends before the next frame where the human needs the robot
        # or robot took action after last need robot atomic action event (ac.frameInit = inf)
        if robotActionEnds < ac.frameInit:
            Rt, Re, R, nextState = self.stepActionInTime(
                action, framesNeededRobot, robotActionEnds, ac)
        
        # robot action ends AFTER the next needRoot atomic action
        else:
            Rt, Re, R, nextState = self.stepActionLate(
                action, framesNeededRobot, robotActionEnds, ac)
                    
        return Rt, Re, R, nextState
   
    def stepActionInTime(self, action, framesNeededRobot, robotActionEnds, ac):

        """
        This method is called when the robot takes an action that does not cause
        the human to wait. Maybe because it ends before the next human-needs-asistance 
        atomic action or because there are no more human-need-assitance moments. 
        The action might be useful for the recipe, in which case it gets no penalty, 
        or incorrect, in which case it does get penalty
        """

        objTaken = cfg.ROBOT_ACTIONS_2_OBJECTS[action]
        actionFrame = self.episode.currentFrame

        # advance episode to the next reward time instant
        self.episode.advanceTo(robotActionEnds)
        self.episode.advanceTo(self.episode.findNextRewardFrame())

        if cfg.ENV_VERBOSE:
            print("\tRobot needs %d frames to complete action" %(framesNeededRobot))
            print("\tRobot action will end at frame: %d" %(robotActionEnds))
            print("\tRobot action will end before next human-need-robot action")
            print("\tRobot is taking the %s " %(objTaken))
            print("\tIs the object necessary for the recipe?: %s" %(str(
                self.episode.isObjectNecessary(objTaken)  )))
            print("\tIs this action possible?: %s" %(self.oit[objTaken] == 0))

        # compute reward
        if self.episode.isObjectNecessary(objTaken) and self.oit[objTaken] == 0:
            timeReward, energyReward, reward = 0,0,0
        
        else:
            if self.episode.currentFrame == ac.frameInit:
                # if requiredObj is not on the table, asr is necessary
                asr = (self.oit[ac.requiredObj] == 0)
                timeReward, energyReward, reward = self.computeReward(
                    actionFrame, ac.frameInit, objTaken, ac.requiredObj, ASR=asr,
                    framesNeeded=framesNeededRobot)
                self.oit[ac.requiredObj] = 1
                
            else:
                timeReward, energyReward, reward = self.computeReward(
                    actionFrame, ac.frameInit, objTaken, 5, ASR=False,
                    framesNeeded=framesNeededRobot)

        # compute nextState and update table 
        #if self.episode.currentFrame == ac.frameInit:
        #    self.episode.advanceTo(ac.frameEnd)

        self.oit[objTaken] = 1
        nextState = self.buildState()

        return timeReward, energyReward, reward, nextState
    
    def stepActionLate(self, action, framesNeededRobot, robotActionEnds, ac):

        """
        This method is called when the robot takes and action that ends after the
        human needs asistance. Human might need to wait or not, depending on previous
        robot actions (current objects in table)
        """
        objTaken = cfg.ROBOT_ACTIONS_2_OBJECTS[action]
        actionFrame = self.episode.currentFrame

        # find the next needRobot atomic action where the human needs to wait
        acWaiting = None
        for ac_i in self.episode.atomicActions:
            if (ac_i.requiresRobot and
                ac_i.frameInit < robotActionEnds and 
                ac_i.frameEnd > self.episode.currentFrame):

                if self.oit[ac_i.requiredObj] == 0:
                    acWaiting = ac_i
                    break
                #else: # we skip the needRobot atomic action
                #    robotActionEnds += ac_i.duration
                #    actionFrame += ac_i.duration

        if cfg.ENV_VERBOSE:
            print("\tRobot needs %d frames to complete action" %(framesNeededRobot))
            print("\tRobot action will end at frame: %d" %(robotActionEnds))
            print("\tRobot action will end after next human-need-robot action")
            print("\tRobot is taking the %s " %(objTaken))
            print("\tIs the object necessary for the recipe?: %s" %(str(
                self.episode.isObjectNecessary(objTaken)  )))
            print("\tIs this action possible?: %s" %(self.oit[objTaken] == 0))
            print("\tDoes the human need to wait?: %s" %(str(acWaiting is not None)))


        # if the human does not need to wait
        if acWaiting is None:

            # move to robotActionEnds frame
            self.episode.advanceTo(robotActionEnds)

            # move to ntri after robot finishes
            self.episode.advanceTo(self.episode.findNextRewardFrame())

            # if objTaken is correct for the recipe and is not on the table already
            if self.episode.isObjectNecessary(objTaken) and self.oit[objTaken] == 0: 
                timeReward, energyReward, reward = 0,0,0
            
            # otherwise, just the lambda*energy penalty
            else:
                timeReward, energyReward, reward = self.computeReward(
                    actionFrame, -1, objTaken, 5, ASR=False,
                    framesNeeded=framesNeededRobot)          

        # the human needs to wait because the requiredObj is not on the table
        else: 

            # compute reward depending on whether the object was correct or not
            asr = False if objTaken == acWaiting.requiredObj else True
            if cfg.ENV_VERBOSE: print("ASR?: %s" %(str(asr)))

            timeReward, energyReward, reward = self.computeReward(
                actionFrame, acWaiting.frameInit, objTaken, acWaiting.requiredObj, 
                ASR=asr, framesNeeded=framesNeededRobot)
            if asr: self.oit[acWaiting.requiredObj] = 1

            # move robot to end of needs to wait ac
            self.episode.advanceTo(acWaiting.frameInit)

        # update iot and compute next state
        self.oit[objTaken] = 1
        nextState = self.buildState()

        return timeReward, energyReward, reward, nextState

    def computeReward(self, actionFrame, waitingFrame, objTook, objNeeded, ASR=False, framesNeeded=None):

        timeReward, energyReward, reward = 0,0,0
        lambda_ = cfg.FACTOR_ENERGY_PENALTY

        # 1 - compute the time it takes the robot to do the action or use the one passed
        if framesNeeded is None:
            framesNeeded = self.computeRobotTime(objTook)

        # 2 - find the amount of frames the human will be waiting until robot comes back
        humanWaits = actionFrame + framesNeeded - waitingFrame
        timeReward = 0 if humanWaits < 0 else humanWaits

        # 3 - add more time and energy penalty if the object that the robot took was not
        # the neccessary object for the human (can pass objNeeded = 5 to avoid adding time)
        humanTime = self.computeHumanTime(objTook)
        if objTook != objNeeded:

            # calculate the time it will take the robot to bring the correct object
            timeReward += self.computeRobotTime(objNeeded)

            # compute the energy penalty using the object it brought
            energyReward = humanTime

        # 4 - add the ASR penalty if neccessary
        if ASR:
            timeReward += cfg.ASR_PENALTY

        # pass argument waitingFrame = -1 to force no time penalty
        if waitingFrame == -1:
            timeReward = 0

        # 5 - If the action is not possible (because the object is already on the table)
        if self.oit[objTook] == 1:
            energyReward += cfg.IMPOSSIBLE_ACTION_PENALTY*humanTime

        # 6 - calculate the combination of Rt + lambda*Re
        reward = timeReward + lambda_*energyReward

        # accumulate the delay for the episode and return
        self.episode.delay += timeReward
        return -timeReward, -energyReward, -reward
    
    def computeRobotTime(self, object):
        
        # the time required by the robot to recover 'object' from fridge
        # Modeled following a gaussian distribution as stated in the ROBIO paper
        # in the paper it also accounts for robot mistakes
        
        beta = cfg.ROBOT_TIME_BETA
        probFail = cfg.ROBOT_PROB_FAILURE
        humanTime = cfg.ROBOT_AVERAGE_DURATIONS[object]
        failed = random.random() < probFail

        mu = beta*(humanTime + failed*(0.5*humanTime))
        std = 0.2*mu

        time = int(random.gauss(mu,std))
        return time

    def computeHumanTime(self, object):

        return cfg.ROBOT_AVERAGE_DURATIONS[object]

    def render(self, mode='human'):
        return
    
    def close(self):
        return
    


if __name__ == '__main__':

    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    sys.path.append(root_path)

    import config2 as cfg
    from utils import readTestTxt

    mode = 'train'
    testFile = "fold1_test.txt"
    
    # env = SimulatedEnv(mode, testFile)
    # state, info = env.reset()
    # env.episode.currentFrame = 550
    # env.oit['milk'] = 1
    # env.oit['nutella'] = 1
    # env.step(3)


    env = SimulatedEnv(mode, testFile)
    
    #for i in range(env.numEpisodes):
    #    state, info = env.reset()
    #    print(env.episode.folderName, env.episode.minDelay)
    
    env.reset()
    for t in count():

        action = int(input("Enter an action: "))
        if action == 6: break
        if action == 7: 
            s,i = env.reset()
            continue

        env.step(int(action))
        print("")
        
        if env.episode.terminated:
            state, info = env.reset()


    


    # dbFolder = os.path.join(cfg.ANNOTATIONS_FOLDER, cfg.DATASET_NAME)
    # allFolders  = sorted(glob.glob(dbFolder+"/*"))
    
    # for f in allFolders:
    
    #     labelsPath = os.path.join(f, cfg.LABELS_FILE_NAME)

    #     e = Episode(f, labelsPath, cfg.ACTIONS_TO_REMOVE_FROM_RECIPES, cfg.ROBOT_NEEDED_FOR_ACTIONS,
    #             cfg.VIDEO_FPS, cfg.ANNOTATIONS_FPS, cfg.ANNOTATIONS_DIM)
    #     #e.visualizeAndSave()
    #     print(e)



    # actionFrame = 528
    # waitingFrame = 528
    # objTook = 'milk'
    # objNeeded = 'milk'

    # rt, re, r = env.computeReward(actionFrame, waitingFrame, objTook, objNeeded, ASR=True)
    # a = 3
    
