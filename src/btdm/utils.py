import numpy as np
import pandas as pd
import matplotlib.pyplot as P
from matplotlib import style
style.use('ggplot')
import random

## Functions
def rolling_sum(a, n=10, normalise=False):
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    if normalise:
        return ret[n - 1:]/n
    else:
        return ret[n - 1:]

def pandify(x, addPerf=False, energyCosts=False):
    if len(x[0])==9:
        cols = ['ep', 'cueState', 'cueID', 'state', 'action', 'logprob', 'reward', 'done', 'totalRewards']
    elif len(x[0])==8:
        cols = ['ep', 'cueState', 'cueID', 'state', 'action', 'reward', 'done', 'totalRewards']
    else:
        raise SystemExit('>> Error in logs!')
    df = pd.DataFrame(x, columns=cols)
    if addPerf:
        comparison = lambda x: x>0 if energyCosts else x==1 # if energyCosts==True, the comparison checks if it is positive, else: if it is equal to 1
        df['perf'] = df.apply(lambda row: 1 if comparison(row['reward']) else 0, axis=1) # convert reward to performance by using comparison function, then adds to a new column: perf
#         df['perf'] = df.apply(lambda row: 1 if row['reward'] == 1 else 0, axis=1) # convert -1 to 0 (failed); new column: perf
    return df
    
def plotSummary(df, n=10):
    ddf = df[df['done']==True] # df for done
    fig, axs = P.subplots(3,1,figsize=(10,5), sharex=True)
    axs[0].plot(rolling_sum(ddf['totalRewards'].tolist(), n));
    axs[0].set_title(f'total rewards (rolling average of {n})')
    axs[1].bar(range(len(ddf)-n+1), rolling_sum(ddf['totalRewards'].tolist(), n))
    axs[1].set_title(f'total reward counts (rolling average of {n})')
    rewCS = np.cumsum(ddf['totalRewards'].tolist()[1:]) #/np.arange(2,len(ddf['totalRewards'].tolist())+1)*100 # percentage
    axs[2].plot(rewCS);
    axs[2].set_title('cum. sum of rewards');
#     axs[2].set_ylabel('percentage (%)');
    axs[2].set_ylim(np.min(rewCS),np.max(rewCS));
    axs[2].set_xlabel('number of episode');
    
def numberOfParams(net):
    params = filter(lambda p: p.requires_grad, net.parameters())
    nParams, sizes = [], []
    for p in params:
        sizes.append(p.size())
        nParams.append(np.prod(p.size()))
    print(f'>> params: {sizes} nParams: {sum(nParams)}')
    
def getSCDData(returnDF = False):
    biodf = pd.read_csv('../data/Data-From-AndrewBarron/SequenceHB.csv')
    # data cleanup (APPYL ONCE)
    biodf.loc[biodf['choice'] == 4, 'actionA'] = 0 # 4 -> 0 option-0: actionA for Analysis
    biodf.loc[biodf['choice'] == 6, 'actionA'] = 1 # 6 -> 1 option-1: actionA for Analysis
    biodf.loc[biodf['choice'] == 2, 'actionA'] = 2 # 2 -> 2 option-2: actionA for Analysis
    biodf['actionA'] = biodf['actionA'].astype('int32')
    bioMean = np.around([biodf[biodf['beeID']==x]['perf'].mean() for x in biodf['beeID'].unique()], 3)
    if not returnDF:
        return bioMean
    else:
        return bioMean, biodf
    
## Lambda Functions
printBioESN = lambda bioMean, esnMean: print(f'>> bioMean: {bioMean.mean():.3f}\tesnMean: {esnMean.mean():.3f}\n'+
                                             f'>> bioMedian: {np.median(bioMean):.3f}\tesnMedian: {np.median(esnMean):.3f}\n'+
                                             f'>> bioMax: {bioMean.max():.3f}\tesnMax: {esnMean.max():.3f}\n'+
                                             f'>> bioMin: {bioMean.min():.3f}\tesnMin: {esnMean.min():.3f}\n'+
                                             f'>> bioStd: {bioMean.std():.3f}\tesnStd: {esnMean.std():.3f}\n')

meanPerf = lambda df: np.around(df[df['done']==True]['perf'].mean(),3) # mean performance, rounded

comment = lambda x: print(f'\033[1m>> comment:\033[0m {x}')

def generateSeq(N = 6, mode='repeat'):
    N = max(2,N)
    if mode in ['repeat', 'SCD', 'sequential']:
        # Sequential repeat
        x = random.randint(0,1)
        R = N//2
        return [x]*R + [(1-x) % 2]*R
    elif mode in ['binary', 'DCD', 'delayed']:
        # The same number does not appear more than twice
        NMBR = [random.randint(0,1) for _ in range(2)]
        for _ in range(N-2): # already decided on the first two
            nmb = random.randint(0,1)
            if (nmb == NMBR[-2]) and (nmb == NMBR[-1]): # if last two are the same, flick it (0->1, 1->0)
                NMBR.append((nmb+1) % 2)
            else: 
                NMBR.append(nmb)
        return NMBR
    else:
        raise ValueError(f">> Received invalid random number generator mode: {mode}")