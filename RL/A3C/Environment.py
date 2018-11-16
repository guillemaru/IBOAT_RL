
import os
import sys
sys.path.append("../../sim/")
from mdp import ContinuousMDP
'''
from settings import ENV, FRAME_SKIP

from PIL import Image
import imageio
'''

class Environment:

    def __init__(self):
        '''
        self.env_no_frame_skip = gym.make(ENV)
        self.env = gym.wrappers.SkipWrapper(FRAME_SKIP)(self.env_no_frame_skip)
        print()
        self.render = False
        self.offset = 0
        self.images = []
        '''
        self.history_duration = 3  # Duration of state history [s]
        self.mdp_step = 1  # Step between each state transition [s]
        self.time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
        self.low_bound = -3.0
        self.high_bound = 3.0
        self.action_size = 3 #number of actions possible
        self.mdp = ContinuousMDP(self.history_duration, self.mdp_step, self.time_step,self.low_bound,self.high_bound)

    def get_state_size(self):
            return  (self.mdp.size) 

    def get_action_size(self):
        return self.action_size

    def reset(self,hdg0,WH):
        s = self.mdp.initializeMDP(hdg0, WH)
        self.mdp.simulator.hyst.reset()
        return s

    def act(self, action,WH, gif=False):

        next_state, reward = self.mdp.transition(action, WH) 
        return next_state, reward 

    def _act(self, action, WH):
        next_state, reward = self.mdp.transition(action, WH) 
        return next_state, reward 


    def get_bounds(self):
        return self.low_bound,self.high_bound #self.env.action_space.low, self.env.action_space.high
