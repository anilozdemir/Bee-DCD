from itertools import chain
import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.utils as utils
import torch.nn.functional as F

from .networks import softMax, NoiseLayer

class DCD_Agent():
    '''
    Custom REINFORCE agent class
    DocString: TODO
    INPUT: env, model and optimiser
    Used for ESN, pre-recorded states
    '''
    def __init__(self, env_get_hiddenStates, env_use_hiddenStates, model, randomSeed = 1, **kwargs):

        self.env_get_hiddenStates = env_get_hiddenStates
        self.model = model

        self.init_hiddenStates()
        if 'contexts' in kwargs:
            self.env = env_use_hiddenStates(hiddenStates = self.hiddenStates, randomSeed = 1, contexts = kwargs['contexts'])
        else:
            self.env = env_use_hiddenStates(hiddenStates = self.hiddenStates, randomSeed = 1)
        
        self.randomSeed = randomSeed
        np.random.seed(randomSeed)
        torch.manual_seed(randomSeed)

    def init_hiddenStates(self):

        env, model = self.env_get_hiddenStates, self.model
        #### Collect ESN state activities ####
        self.hiddenStates = torch.zeros((env.action_space.n), model.nReservoir)
        for i in range(env.action_space.n):
            # reset env and the model
            state =  env.reset()
            model.reset()
            # get the first state, (looping through 3 options)
            state0 = np.stack([[0,0], env.target_colours[i] , [0,0]]).flatten()
            # feed the first state to the model
            model(torch.Tensor([state0])); 
            # second state is static: always the same
            state1 = env.target_colours.flatten()
            # feed the second state to the model
            model(torch.Tensor([state1])); 
            # collect the last state of the reservoir
            self.hiddenStates[i] = model.recursiveState[-1]

    def init_readout_threshold_optim(self, lR, quantile, nHiddenLayer = None, hiddenLayerSize = None, noise = None, temp = 1, thrGrad = False, SPARCE = True, **kwargs):

        # np.random.seed(self.randomSeed)
        # torch.manual_seed(self.randomSeed)

        if nHiddenLayer == None    : nHiddenLayer = 0 # if it is None, it has to be overwritten
        if hiddenLayerSize == None : hiddenLayerSize = 0 # if it is None, it has to be overwritten
        if noise == None           : noise = 0 # if it is None, it has to be overwritten
        
        self.nHiddenLayer    = nHiddenLayer
        self.hiddenLayerSize = hiddenLayerSize
        self.SPARCE          = SPARCE

        model = self.model

        #### ESN-ReadOut: Wout #### 
        model.distr = kwargs.get('distr', 'uniform')
        model.phi   = kwargs.get('phi'  , 1.0)

        if nHiddenLayer == None or nHiddenLayer == 0:
            nWoutInput = model.nReservoir
        elif nHiddenLayer < 0:
            raise ValueError('Wrong number of hidden layers... It has to be 0 or a positive number')
        else:
            if hiddenLayerSize == None or hiddenLayerSize == 0:
                raise ValueError('Wrong hidden layer size!... It has to be 0 or a positive number')
            nWoutInput = hiddenLayerSize

        if model.distr == 'uniform':
            wout = np.random.uniform(-model.phi, model.phi, [model.nOutput, nWoutInput]) / nWoutInput
        elif model.distr.lower() == 'normal' or  model.distr.lower() == 'gaussian':
            wout = np.random.normal(0, model.phi, [model.nOutput, nWoutInput])
        else:
            raise SystemExit('>> Error in init. distribution!')
    
        wout = torch.tensor(wout, dtype = model.dtype).to(model.device)
        self.Wout = torch.nn.Parameter(wout, requires_grad = True)
        
        ####  Init SPARCE Thresholds ####
        self.thrGrad    = thrGrad
        self.thr_fixed  = torch.nn.Parameter(torch.quantile(torch.abs(self.hiddenStates), quantile, dim=0).unsqueeze_(0), requires_grad = False)  # update 2 # calculate quantile values (one for each neuron, over nSample samples)
        # params          = [{'params': self.Wout, 'lr': lR}]
    
        #### Update Model Params ####
        model.quantile = quantile
        model.thrGrad  = self.thrGrad

        #### Activation function ####
        self.activation = softMax(temp = temp)

        #### Noise Layer ####
        self.noise = noise
        self.noiseLayer = NoiseLayer(noise)

        class outputLayer(nn.Module):
            def __init__(self, Wout, noiseLayer, activation):
                super().__init__()
                self.Wout = Wout
                self.noiseLayer = NoiseLayer(noise)
                self.activation = activation

            def forward(self,x):
                out = F.linear(x, self.Wout)  # pass through read-out weights 
                out = self.noiseLayer(out)    # pass through the noise layer
                out = self.activation(out)    # pass through the softmax activation
                return out

        self.bias = kwargs.get('bias', True)
        #### Hidden Layers ####
        self.outputLayer = outputLayer(self.Wout, self.noiseLayer, self.activation)

        if nHiddenLayer == 0:
            params = [{'params': self.outputLayer.parameters(), 'lr': lR}]
            self.forwardLayer = self.outputLayer

        if nHiddenLayer >= 1:
            self.resToHidden = nn.Sequential(nn.Linear(model.nReservoir, hiddenLayerSize, bias=self.bias), NoiseLayer(noise), nn.ReLU())
            self.oneHiddenLayer = nn.Sequential(self.resToHidden, self.outputLayer)
            params = [{'params': self.oneHiddenLayer.parameters(), 'lr': lR}]
            self.forwardLayer = self.oneHiddenLayer

        if nHiddenLayer > 1:
            listHiddenLayers = [[nn.Linear(hiddenLayerSize, hiddenLayerSize, bias=self.bias), NoiseLayer(noise), nn.ReLU()] for _ in range(nHiddenLayer-1)]
            self.hiddenLayer = nn.Sequential(*list(chain(*listHiddenLayers)))
        
            self.moreThanOneHiddenLayer = nn.Sequential(self.resToHidden,self.hiddenLayer, self.outputLayer)
            params = [{'params': self.moreThanOneHiddenLayer.parameters(), 'lr': lR}]

            self.forwardLayer = self.moreThanOneHiddenLayer
        
        #### Update Learning Weights (thr_learn) ####
        if self.thrGrad:
            self.thr_learn = torch.nn.Parameter(torch.zeros((*self.thr_fixed.shape)), requires_grad = True).to(model.device)  # update
            lR_divide_factor = kwargs.get('lR_divide_factor', 10)
            lR_Thr = lR / lR_divide_factor
            params.append({'params':  self.thr_learn, 'lr': lR_Thr})
            # update model with the SPARCE params
            model.lR_Thr   = lR_Thr
        
        self.optim = torch.optim.Adam(params) 

    def model_predict(self, x):

        # Add noise to the reservoir representation:
        x = self.noiseLayer(x)

        if self.SPARCE:
            # given input_state (x) return actions
            thr = self.thr_fixed if not self.thrGrad else self.thr_learn + self.thr_fixed  # check if thrGrad is True or not (if not, use only thr_fixed, else, use both)
            v   = torch.sign(x) * F.relu(torch.abs(x) - thr) # pass through thresholds
        else:
            v = x.unsqueeze_(0)

        out = self.forwardLayer(v)
        probs = out[0]
        ### select action and move on
        action = torch.multinomial(probs, 1).item()
        return probs, action


    def train(self, nEpoch=100, returnDF=False, silent=False):
        
        # Note: this function is written considering the model is in CPU!!! 
        
        env  = self.env

        # reverse assignment
        optimizer = self.optim
        
        total_rewards = 0
        total_perf    = 0
        logs = []

        for ep in trange(nEpoch, disable=silent):
            state   = env.reset()
            probs, action = self.model_predict(state)
            log_prob = torch.log(probs[action])
            next_state, reward, done, _ = env.step(action)            
            total_rewards += reward

            perf = 0 if reward == -1 else reward
            total_perf += perf

            logs.append([ep, env.cueID, action, np.round(log_prob.cpu().detach().numpy(),3), reward, total_rewards, perf, total_perf])                
            # update time, at every epoch
            policy_gradient = [-log_prob * reward]
            loss = torch.stack(policy_gradient).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Performance operations
        cols = ['ep', 'cueID', 'action', 'logprob', 'reward', 'totalRewards', 'perf', 'totalPerf']
        self.logs     = logs
        self.df       = pd.DataFrame(logs, columns = cols)
        self.perf     = np.around(self.df['perf'].mean(),3)
        self.nParam   = sum([np.prod(param.data.shape) for _, param in self.model.named_parameters() if param.requires_grad])
        
        if returnDF:
            return self.df
    
    def evaluate(self, nTrial=30, returnDF=False, silent=False):
        env  = self.env
        
        total_rewards = 0
        total_perf    = 0
        logs = []

        for ep in trange(nTrial, disable=silent):
            state = env.reset()
            probs, action = self.model_predict(state)
            log_prob = torch.log(probs[action])
            next_state, reward, done, _   = env.step(action)            
            total_rewards += reward
            perf = 0 if reward == -1 else reward
            total_perf += perf

            logs.append([ep, env.cueID, action, np.round(log_prob.cpu().detach().numpy(),3), reward, total_rewards, perf, total_perf])   
            
        # Performance operations
        cols = ['ep', 'cueID', 'action', 'logprob', 'reward', 'totalRewards', 'perf', 'totalPerf']
        self.eval_logs = logs
        self.eval_df   = pd.DataFrame(logs, columns = cols)
        self.eval_perf = np.around(self.eval_df['perf'].mean(),3)
        
        if returnDF:
            return self.eval_perf