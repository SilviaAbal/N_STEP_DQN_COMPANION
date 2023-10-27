import random
from collections import namedtuple, deque


Transition = namedtuple('Transition',
                        ('pHidden', 'state', 'action', 'tHidden', 'next_state', 'reward'))


class ReplayMemory(object):

    """
    This class implements a first in first out buffer with maximum
    capacity "capacity". It is possible to sample 'batch_size' 
    elements (unique elements) from the queue
    """
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        return
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        return
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

if __name__ == '__main__':

    t1 = Transition(-1, 0, 'a', -2, 1, -1)
    t2 = Transition(-2, 1, 'b', -2, 2, -1)
    t3 = Transition(-3, 1, 'c', -3, 2,  0)

    rm = ReplayMemory(3)
    rm.push(*t1)
    rm.push(*t2)
    rm.push(*t3)

    # now each transition is there
    print(rm.sample(3))

    # now this push has removed t1 from the replay memory
    rm.push(*t3)
    print(rm.sample(3))


