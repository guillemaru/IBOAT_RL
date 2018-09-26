
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
        self.history_duration = 6  # Duration of state history [s]
        self.mdp_step = 1  # Step between each state transition [s]
        self.time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
        self.low_bound = -3.0
        self.high_bound = 3.0
        self.action_size = 1
        self.mdp = ContinuousMDP(self.history_duration, self.mdp_step, self.time_step,self.low_bound,self.high_bound)

    def get_state_size(self):
        #try:
            return  (self.mdp.size) #(self.env.observation_space.n, )
        #except AttributeError:
        #    return list(self.env.observation_space.shape)

    def get_action_size(self):
        return self.action_size #self.env.action_space.n

    #def set_render(self, render):
    #    self.render = render

    def reset(self,hdg0,WH):
        s = self.mdp.initializeMDP(hdg0, WH)
        self.mdp.simulator.hyst.reset()
        return s
        #return self.env.reset()

    def act(self, action, gif=False):
        '''
        if gif:
            return self._act_gif(action)
        else:
            return self._act(action)
        '''
        next_state, reward = self.mdp.transition(action, WH) 
        return next_state, reward 

    def _act(self, action):
        '''
        if self.render:
            self.env.render()
        return self.env.step(action)
        '''
        next_state, reward = self.mdp.transition(action, WH) #self.env.step(action)
        return next_state, reward 
    '''
    def _act_gif(self, action):
        r = 0
        i, done = 0, False
        while i < (FRAME_SKIP + 1) and not done:
            if self.render:
                self.env_no_frame_skip.render()

            # Save image
            img = Image.fromarray(self.env.render(mode='rgb_array'))
            img.save('tmp.png')
            self.images.append(imageio.imread('tmp.png'))

            s_, r_tmp, done, info = self.env_no_frame_skip.step(action)
            r += r_tmp
            i += 1
        return s_, r, done, info

    def save_gif(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        imageio.mimsave(path, self.images, duration=1)
        self.images = []
    
    def close(self):
        self.env.close()
    '''

    def get_bounds(self):
        return self.low_bound,self.high_bound #self.env.action_space.low, self.env.action_space.high
