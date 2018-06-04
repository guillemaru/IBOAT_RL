
import tensorflow as tf
import numpy as np

#import gym
#from gym import wrappers
import random

from QNetwork import Network

from Environment import Environment
from ExperienceBuffer import ExperienceBuffer

from Displayer import DISPLAYER
import parameters

import sys
sys.path.append("../sim/")
from Simulator import TORAD
from mdp import ContinuousMDP
from environment import wind



class Agent:

    def __init__(self, sess):
        print("Initializing the agent...")

        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()*2 #60 velocities and 60 incidences
        self.action_size = self.env.get_action_size()
        self.low_bound, self.high_bound = self.env.get_bounds()

        self.buffer = ExperienceBuffer()

        print("Creation of the actor-critic network")
        self.network = Network(self.state_size, self.action_size,
                               self.low_bound, self.high_bound)

        self.sess.run(tf.global_variables_initializer())
        DISPLAYER.reset()

    def run(self):

        self.total_steps = 0

        hdg0_rand_vec=(0,2,4,6,8,10,13,15,17,20)
        '''
        WIND CONDITIONS
        '''
        mean = 45 * TORAD
        std = 0 * TORAD
        wind_samples = 10
        w = wind(mean=mean, std=std, samples = wind_samples)
        WH = w.generateWind()

        for ep in range(1, parameters.TRAINING_STEPS+1):

            episode_reward = 0
            episode_step = 0
            #done = False

            # Initialize exploration noise process
            noise_process = np.zeros(self.action_size)
            noise_scale = (parameters.NOISE_SCALE_INIT *
                           parameters.NOISE_DECAY**ep) * \
                (self.high_bound - self.low_bound)

            # Initial state
            WH = w.generateWind()
            hdg0_rand = random.sample(hdg0_rand_vec, 1)[0]
            hdg0 = hdg0_rand * TORAD * np.ones(10)
            s = self.env.reset(hdg0,WH)

            #render = (ep % parameters.RENDER_FREQ == 0 and parameters.DISPLAY)
            #self.env.set_render(render)

            while episode_step < parameters.MAX_EPISODE_STEPS: #and not done:

                WH = np.random.uniform(mean - std, mean + std, size=wind_samples)

                # choose action based on deterministic policy
                s = np.reshape([s[0,:], s[1,:]], [self.state_size,1])
                a, = self.sess.run(self.network.actions,
                                   feed_dict={self.network.state_ph: s[None]})

                # add temporally-correlated exploration noise to action
                # (using an Ornstein-Uhlenbeck process)
                noise_process = parameters.EXPLO_THETA * \
                    (parameters.EXPLO_MU - noise_process) + \
                    parameters.EXPLO_SIGMA * np.random.randn(self.action_size)

                a += noise_scale * noise_process
                #to respect the bounds:
                if a>3:
                    a=3
                if a<-3:
                    a=-3

                s_, r  = self.env.act(a,WH) #, done, info
                episode_reward += r

                if a==3 or a==-3:
                    a=[a]

                self.buffer.add((s, np.reshape(a, [1,1] ), r, np.reshape(s_, [self.state_size,1]), 0.0 if episode_step<parameters.MAX_EPISODE_STEPS-1 else 1.0)) #, 0.0 if done else 1.0

                # update network weights to fit a minibatch of experience
                if self.total_steps % parameters.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= parameters.BATCH_SIZE:

                    minibatch = self.buffer.sample()

                    _, _,actor_loss,critic_loss = self.sess.run([self.network.critic_train_op, self.network.actor_train_op,self.network.actor_loss,self.network.critic_loss],
                                         feed_dict={
                        self.network.state_ph: np.asarray([elem[0] for elem in minibatch]),
                        self.network.action_ph: np.asarray([elem[1] for elem in minibatch]),
                        self.network.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                        self.network.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                        self.network.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch])})

                    # update target networks
                    _ = self.sess.run(self.network.update_slow_targets_op)

                s = s_
                episode_step += 1
                self.total_steps += 1

            if ep % parameters.DISP_EP_REWARD_FREQ == 0:
                print('Episode %2i, initial heading: %7.3f, Reward: %7.3f, Actor loss: %7.3f, Critic loss: %7.3f, Final noise scale: %7.3f' %
                      (ep, hdg0[0]*(1/TORAD), episode_reward, actor_loss, critic_loss, noise_scale))
            DISPLAYER.add_reward(episode_reward)


    def play(self, number_run):
        print("Playing for", number_run, "runs")

        #self.env.set_render(True)
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                #done = False
                episode_step=0
                while episode_step < parameters.MAX_EPISODE_STEPS: #not done:

                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: s[None]})

                    s, r, done, info = self.env.act(a)
                    episode_reward += r
                    episode_step += 1
                
                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            #self.env.set_render(False)
            print("End of the demo")
            #self.env.close()

    #def close(self):
    #    self.env.close()
