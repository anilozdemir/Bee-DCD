import copy
import random
import numpy as np
import gym

class DCD(gym.Env):
    INFO   = 'This env. has 2 steps. The task is to associate cue and action.'
    def __init__(self, randomSeed = 1, permute=False, contexts = np.array([[1,0], [0,1]])): 
        super(DCD, self).__init__()

        nActions               = 3 # blue, yellow, green
        self.action_space      = gym.spaces.Discrete(nActions)
        self.observation_space = gym.spaces.Box(low=np.array([0]*6), high=np.array([1]*6), dtype=np.float32) # Real-valued    
        self.target_colours    = np.array([[1,0], [0,1], [1,1]]) # blue, yellow, green
        self.context_colours   = contexts
        self.BLUE, self.YELLOW, self.GREEN = 0, 1, 2
        self.permute           = permute
        self.cueID_list        =  []
        self.reset()
    
    def register_env(self, _):
        pass

    def reset(self):
        self.timestep = 0
        
        cueID = random.randint(0,1)
        if len(self.cueID_list) > 1:
            if (cueID == self.cueID_list[-2]) and (cueID == self.cueID_list[-1]): # if last two are the same, flick it (0->1, 1->0)
                cueID = (cueID+1) % 2 # if last two are the same, flip
        
        self.cueID_list.append(cueID)
        self.context(cueID)                     # register the cue
        return self.lastState                   # return what is actually seen, the first time!

    def context(self, cueID):
        self.cueID     = cueID
        self.cueState  = np.stack([[0,0], self.target_colours[cueID] , [0,0]]).flatten()  # meaning of what is seen (black for blue)
        self.lastState = np.stack([[0,0], self.context_colours[cueID], [0,0]]).flatten() # what is actually seen (e.g. black)

    def step(self, action):
        done   = False
        reward = 0
        if self.timestep != 0:
            state = 'landed'
            done  = True
            if action == self.cueID:
                reward = 1
            elif action == self.GREEN:
                reward = 0
            else:
                reward = -1
        else: # if time step is 0; no reward, no action takes place
            if self.permute:
                state  = np.random.permutation(self.target_colours).flatten() # permuted colours
            else: 
                state  = self.target_colours.flatten() # state of the random look             
            self.lastState = copy.copy(state)   # copy the state to lastState 
        self.timestep += 1 # increase the time step
        return state, reward, done, self.INFO

    def render(self):
        pass

    def close(self):
        pass 

class DCD_SingleStep(gym.Env):
    INFO   = 'This env. has 1 step. The task is to associate cue and action.'
    def __init__(self, randomSeed = 1, contexts = np.array([[1,0], [0,1]])): 
        super().__init__()

        # np.random.seed(randomSeed)

        nActions               = 3 # blue, yellow, green
        self.action_space      = gym.spaces.Discrete(nActions)
        self.observation_space = gym.spaces.Box(low=np.array([0]*6), high=np.array([1]*6), dtype=np.float32) # Real-valued    
        self.target_colours    = np.array([[1,0], [0,1], [1,1]]) # blue, yellow, green
        self.context_colours   = contexts
        self.BLUE, self.YELLOW, self.GREEN = 0, 1, 2
        self.cueID_list        =  []
        # self.reset()
    
    def register_env(self, _):
        pass

    def reset(self):
        cueID = random.randint(0,1)
        if len(self.cueID_list) > 1:
            if (cueID == self.cueID_list[-2]) and (cueID == self.cueID_list[-1]): # if last two are the same, flick it (0->1, 1->0)
                cueID = (cueID+1) % 2 # if last two are the same, flip
        
        self.cueID     = cueID
        self.cueID_list.append(cueID)
        self.lastState = np.stack([[0,0], self.context_colours[cueID], [0,0]]).flatten() # what is actually seen (e.g. black)
        return self.lastState                   # return what is actually seen, the first time!

    def step(self, action):
        done   = True
        state = 'landed'
        if action == self.cueID:
            reward = 1
        elif action == self.GREEN:
            reward = 0
        else:
            reward = -1        
        return state, reward, done, self.INFO

    def render(self):
        pass

    def close(self):
        pass 
    

class DCD_ESN_SingleStep(DCD_SingleStep):
    ''' Inherits DCD_SingleStep, the input are the state activities of the last timestep'''
    
    INFO   = 'This env. has 1 steps. The task is to associate cue and action. Includes ESN state activities'

    def __init__(self, hiddenStates, randomSeed = 1, contexts = np.array([[1,0], [0,1]])): 
        ''' 
        hiddenStates: Provide Echo States (last timestep only)
        '''
        
        # np.random.seed(randomSeed)

        self.hiddenStates = hiddenStates
        
        super().__init__(randomSeed = randomSeed, contexts = contexts)

        self.observation_space = gym.spaces.Box(low=np.array([-1]*hiddenStates.shape[1]), high=np.array([1]*hiddenStates.shape[1]), dtype=np.float32)

    def reset(self):
        cueID = random.randint(0,1)
        if len(self.cueID_list) > 1:
            if (cueID == self.cueID_list[-2]) and (cueID == self.cueID_list[-1]): # if last two are the same, flick it (0->1, 1->0)
                cueID = (cueID+1) % 2 # if last two are the same, flip
        
        self.cueID = cueID
        self.cueID_list.append(cueID)
        return self.hiddenStates[cueID]    
    