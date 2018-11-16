
import sys

sys.path.append("../../sim/")

import os

from mdp import ContinuousMDP #choose mdp instead to use the simplified simulator

class Environment:

    def __init__(self):
        self.history_duration = 6  # Duration of state history [s]
        self.mdp_step = 1  # Step between each state transition [s]
        self.time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
        self.low_bound = -1.5
        self.high_bound = 1.5
        self.action_size = 1
        self.mdp = ContinuousMDP(self.history_duration, self.mdp_step, self.time_step,self.low_bound,self.high_bound)
    def get_state_size(self):
        return (self.mdp.size)

    def get_action_size(self):
        return self.action_size

    def get_bounds(self):
        return self.low_bound,self.high_bound 

    def reset(self,hdg0,WH):
        s = self.mdp.initializeMDP(hdg0, WH)
        self.mdp.simulator.hyst.reset()
        return s

    def act(self, action,WH):
        next_state, reward = self.mdp.transition(action, WH)
        return next_state, reward 
