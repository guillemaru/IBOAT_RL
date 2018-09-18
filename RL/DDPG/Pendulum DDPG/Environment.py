
import os
import gym
from gym import wrappers
from parameters import ENV, FRAME_SKIP


class Environment:

    def __init__(self):

        self.env_no_frame_skip = gym.make(ENV)
        #self.env = self.env_no_frame_skip
        self.env = gym.wrappers.SkipWrapper(FRAME_SKIP)(self.env_no_frame_skip)
        self.render = False
        self.images = []

    def get_state_size(self):
        try:
            return (self.env.observation_space.n, )
        except AttributeError:
            return list(self.env.observation_space.shape)

    def get_action_size(self):
        try:
            return self.env.action_space.n
        except AttributeError:
            return self.env.action_space.shape[0]

    def get_bounds(self):
        return self.env.action_space.low, self.env.action_space.high

    def set_render(self, render):
        self.render = render

    def reset(self):
        return self.env.reset()

    def act(self, action):
        if self.render:
            self.env.render()
        return self.env.step(action)

    def close(self):
        self.env.close()
