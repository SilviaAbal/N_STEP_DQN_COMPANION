import matplotlib.pyplot as plt


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

    # compute the average stats over each episode
    eps = [sum(epi) / len(epi) for epi in st.get('epsilon', epoch)]
    bl  = [sum(epi) / len(epi) for epi in st.get('batch_loss', epoch)]
    aq  = [sum(epi) / len(epi) for epi in st.get('batch_avgMaxQ', epoch)]

    # for the instant reward, we are going to compute the cumulative instant reward
    ir  = [sum(epi) for epi in st.get('instant_reward', epoch)]

    x = list(st.data[epoch].keys())
    # now we can just plot and save the figures
    plotHelper( x, eps, 'episode#', 'epsilon', 'epoch %d, exploring  prob' %epoch, p+'eps.png')
    plotHelper( x, bl, 'episode#', 'avgEpisodeLoss', 'epoch %d, avg loss' %epoch, p+'bl.png')
    plotHelper( x, aq, 'episode#', 'avgMaxQ', 'epoch %d, avg max Q' %epoch, p+'aq.png')
    plotHelper( x, ir, 'episode#', 'cumReward', 'epoch %d, cumulative reward' %epoch, p+'r.png')
    return

def plotHelper( xData, yData, xLabel, yLabel, title, filePath):


    fig, ax = plt.subplots()

    ax.plot(xData, yData)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)

    fig.savefig(filePath)
    return
