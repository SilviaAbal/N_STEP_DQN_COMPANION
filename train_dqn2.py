"""
This is the code that needs to be launched to train the model
"""
def loadConfiguration():
    
    print("Loading config...")
    cfg = None
    
    print("\tDone!")
    return cfg

def initExperiment(cfg):
    
    print("Initializing experiment variables...")
    exp = {}

    print("\tDone!")
    return exp

def runExperiment(cfg, exp):

    print("Starting experiment")
    results = {}

    print("\tDone!")
    return results

def saveAndVisualize(results):
    
    print("Saving and visualizing results")
    
    print("\tDone!")
    return

if( __name__ == '__main__' ):

    # 1 - Load the experiment config
    cfg = loadConfiguration()

    # 2 - Init experiment env, models, opt, loss, stat objects, etc.
    exp = initExperiment(cfg)

    # 3 - Run the training/validation process
    results = runExperiment(cfg, exp)

    # 4 - Save and visualize results
    saveAndVisualize(results)