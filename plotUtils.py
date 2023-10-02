import matplotlib.pyplot as plt


def plotEpisodeStats( st, epoch, epi):
    
    p = 'figs/episodes/'

    eps = st.get('epsilon', epoch, epi)
    bl  = st.get('batch_loss', epoch, epi)
    aq  = st.get('batch_avgMaxQ', epoch, epi)
    ir  = st.get('instant_reward', epoch, epi)

    plotHelper( eps, 't', 'current eps', 'epoch: %d, epi: %d' %(epoch, epi),p+'eps.png')
    plotHelper( bl, 't', 'batch_loss', 'epoch: %d, epi: %d' %(epoch, epi),  p+'bl.png')
    plotHelper( aq, 't', 'batch_avgMaxQ', 'epoch: %d, epi: %d' %(epoch, epi),  p+'aq.png')
    plotHelper( ir, 't', 'instant_reward', 'epoch: %d, epi: %d' %(epoch, epi),  p+'ir.png')
    return

def plotEpochStats( st, epoch ):

    p = 'figs/epochs/'+str(epoch)+'_'

    # compute the average stats over each episode
    eps = [sum(epi) / len(epi) for epi in st.get('epsilon', epoch)]
    bl  = [sum(epi) / len(epi) for epi in st.get('batch_loss', epoch)]
    aq  = [sum(epi) / len(epi) for epi in st.get('batch_avgMaxQ', epoch)]

    # for the instant reward, we are going to compute the cumulative instant reward
    ir  = [sum(epi) for epi in st.get('instant_reward', epoch)]

    # now we can just plot and save the figures
    plotHelper( eps, 'episode#', 'epsilon', 'epoch %d, exploring  prob', p+'eps.png')
    plotHelper( bl, 'episode#', 'avgEpisodeLoss', 'epoch %d, avg loss', p+'bl.png')
    plotHelper( aq, 'episode#', 'avgMaxQ', 'epoch %d, avg max Q', p+'aq.png')
    plotHelper( ir, 'episode#', 'cumReward', 'epoch %d, cumulative reward', p+'r.png')
    return

def plotHelper( xData, xLabel, yLabel, title, filePath ):

    fig, ax = plt.subplots()

    ax.plot(range(len(xData)), xData)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title(title)

    fig.savefig(filePath)
    return
