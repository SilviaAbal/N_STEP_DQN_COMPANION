import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

def plotEpisodeStats( st, epoch, epi):
    
    p = 'figs/episodes/'

    eps = st.get('epsilon', epoch, epi)
    bl  = st.get('batch_loss', epoch, epi)
    aq  = st.get('batch_avgMaxQ', epoch, epi)
    ir  = st.get('instant_reward', epoch, epi)

    tPoints = list(st.data[epoch][epi].keys())

    plotHelper( tPoints, eps, 't', 'current eps', 'epoch: %d, epi: %d' %(epoch, epi),p+'eps.png')
    plotHelper( tPoints, bl, 't', 'batch_loss', 'epoch: %d, epi: %d' %(epoch, epi),  p+'bl.png')
    plotHelper( tPoints, aq, 't', 'batch_avgMaxQ', 'epoch: %d, epi: %d' %(epoch, epi),  p+'aq.png')
    plotHelper( tPoints, ir, 't', 'instant_reward', 'epoch: %d, epi: %d' %(epoch, epi),  p+'ir.png')
    return

def plotEpochStats( st, epoch ):

    p = 'figs/epochs/'+str(epoch)+'_'

    # compute some average stats over each episode
    eps = [sum(epi) / len(epi) for epi in st.get('epsilon', epoch)]
    bl  = [sum(epi) / len(epi) for epi in st.get('batch_loss', epoch)]
    aq  = [sum(epi) / len(epi) for epi in st.get('batch_avgMaxQ', epoch)]

    # for the episode delay, we don't do the avg, just keep the last recorded delay
    ed =  [epi[-1] for epi in st.get('episode_delay', epoch)]

    # for the instant reward, we are going to compute the cumulative instant reward
    ir  = [sum(epi) for epi in st.get('instant_reward', epoch)]

    # for the actions taken per episode, we plot the amount of them != 5 ('do nothing')
    at = []
    for episode in st.get('actions_taken', epoch):
        epiActions = [e for t in episode for e in list(t.values())]
        at.append(sum(1 for e in epiActions if e != 5))

    x = list(st.data[epoch].keys())
    # now we can just plot and save the figures
    plotHelper( x, eps, 'episode#', 'epsilon', 'epoch %d, exploring  prob' %epoch, p+'eps.png')
    plotHelper( x, bl, 'episode#', 'avgEpisodeLoss', 'epoch %d, avg loss' %epoch, p+'bl.png')
    plotHelper( x, aq, 'episode#', 'avgMaxQ', 'epoch %d, avg max Q' %epoch, p+'aq.png')
    plotHelper( x, ir, 'episode#', 'cumReward', 'epoch %d, cumulative reward' %epoch, p+'r.png', True)
    plotHelper( x, ed, 'episode#', 'delay (frames)', 'epoch %d, delay per recipe' %epoch, p+'delay.png', True)
    plotHelper( x, at, 'episode#', 'actions taken (!=5)', 'epoch %d, actions per recipe' %epoch, p+'numA.png', True)
    return

def plotHelper( xData, yData, xLabel, yLabel, title, filePath, addLPF=False):

    

    fig, ax = plt.subplots()
    ax.plot(xData, yData, 'b')

    if len(yData) > 15 and addLPF:
        order = 4
        cutoff_freq = 0.01
        b, a = signal.butter(order, cutoff_freq, btype='low')
        filt_yData = signal.filtfilt(b, a, yData)
        ax.plot(xData, filt_yData, 'r')

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)

    fig.savefig(filePath)
    return

def plotHelper2( yData1, yData2, xLabel, yLabel, title, legend, recipeNames, filePath):

    xData = range(len(yData1))
    fig, ax = plt.subplots(figsize=(21,4))

    label1 = legend[0]+' = %.2f' %(np.mean(yData1))
    label2 = legend[1]+' = %.2f' %(np.mean(yData2))

    ax.plot(xData, yData1, 'b', label=label1)
    ax.plot(xData, yData2, 'r', label=label2)
    ax.legend()

    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)
    ax.set_xticks(xData)
    ax.set_xticklabels(recipeNames, rotation=90)

    fig.savefig(filePath, bbox_inches="tight")
    return