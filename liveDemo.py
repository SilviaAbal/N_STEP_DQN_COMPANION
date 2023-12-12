"""
This script contains the main code to run the companion demonstrator
"""
from concurrent.futures import ThreadPoolExecutor, wait
import time
import random

# common
class SharedInfo():

    def __init__(self):

        self.currentTime = 0
        self.a = [None, None, None, None]
        self.finished = [False, False]
        return
    
    def run(self):
        print("starting SharedInfo.run()")
        while sum(self.finished) < 2:
            print("sharedInfo run step")
            print(self.a)
            time.sleep(2)

        print("finishing SharedInfo.run()")
        return

# video feeds
class GlassesFeed():

    def __init__(self, sharedInfo):

        self.sharedInfo = sharedInfo
        return
    
    def run(self):
        print("starting GlassesFeed.run()")

        for i in range(10):

            # generate random number
            r = random.randint(10,20)
            idx = random.randint(0,3)
            #print("glassesFeed step: %d" %r)

            self.sharedInfo.a[idx] = r
            time.sleep(1)

        self.sharedInfo.finished[0] = True
        print("finishing GlassesFeed.run()")
        return

class RobotFeed():

    def __init__(self, sharedInfo):

        self.sharedInfo = sharedInfo
        return
    
    def run(self):
        print("starting RobotFeed.run()")
        for i in range(10):

            # generate random number
            r = random.randint(21,30)
            idx = random.randint(0,3)
            #print("robotFeed step: %d" %r)
            self.sharedInfo.a[idx] = r
            time.sleep(1)

        self.sharedInfo.finished[1] = True
        print("finishing RobotFeed.run()")
        return
    
# deep models
class RDM():

    def __init__(self):

        return  
class ASR():

    def __init__(self):

        return 
class IntentionsDetector():

    def __init__(self):

        return
    


if __name__ == '__main__':

    sharedInfo = SharedInfo()
    glassesFeed = GlassesFeed(sharedInfo)
    robotFeed = RobotFeed(sharedInfo)

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        
        future1 = executor.submit(glassesFeed.run)
        future2 = executor.submit(robotFeed.run)
        future3 = executor.submit(sharedInfo.run)

        # Wait for all tasks to complete
        wait([future1, future2, future3])
