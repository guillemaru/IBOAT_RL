# ¡¡¡¡¡DDPG!!!!

import tensorflow as tf
import numpy as np

import random
import math

from QNetwork import Network
from Environment import Environment
from ExperienceBuffer import ExperienceBuffer
from Displayer import DISPLAYER
import parameters

import sys
sys.path.append("../../../sim/")
from Simulator import TORAD 
from mdp import ContinuousMDP
#change to Simulator and mdp to train with a simplified simulator 
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
        #self.load("NetworkParam_best_ThirdSemester/FinalParam") #get the best parameters to start the training
        self.total_steps = 0

        '''
        WIND CONDITIONS
        '''
        mean = 45 * TORAD
        std = 0.1 * TORAD
        wind_samples = 10
        w = wind(mean=mean, std=std, samples = wind_samples)
        WH = w.generateWind()

        for ep in range(1, parameters.TRAINING_STEPS+1):

            episode_reward = 0
            episode_step = 0
            nearlyDone=0
            done=False

            # Initialize exploration noise process
            noise_process = np.zeros(self.action_size)
            noise_scale = (parameters.NOISE_SCALE_INIT *
                           parameters.NOISE_DECAY**ep) * \
                (self.high_bound - self.low_bound)

            # Initial state
            w = wind(mean=mean, std=std, samples = wind_samples)
            WH = w.generateWind()
            hdg0_rand = random.uniform(6,13) 
            hdg0 = hdg0_rand * TORAD * np.ones(10)
            s = self.env.reset(hdg0,WH)
            
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
                a = np.clip(a, self.low_bound, self.high_bound)
                
                s_, v  = self.env.act(a,WH)
                
                #reward  assignation algorithm
                if episode_step==1:
                    r=0
                #elif s[int(self.state_size/2-2)]>(13*TORAD) and s[int(self.state_size/2-2)]<(15*TORAD) and v>0.63 and v<0.67 and a<0:
                #    r=0.1
                else:
                    if v<=0.69:
                        r=0
                        nearlyDone = 0
                    elif v>0.69 and v<=0.75:
                        r=0.00001
                        nearlyDone = 0
                    elif v>0.75 and v<=0.8:
                        r=0.01
                        nearlyDone = 0
                    elif v>0.80:
                        r=0.1
                        if nearlyDone>=3:
                            r=1
                            done = True
                        elif nearlyDone==2:
                            r=0.8
                        elif nearlyDone==1:
                            r=0.25
                        nearlyDone=nearlyDone+1
                    else:
                        r=0
                        nearlyDone = False

                episode_reward += r

                self.buffer.add((s, np.reshape(a, [1,1] ), r, np.reshape(s_, [self.state_size,1]), 0.0 if episode_step<parameters.MAX_EPISODE_STEPS-1 else 1.0)) #, 0.0 if done else 1.0

                # update network weights to fit a minibatch of experience
                if self.total_steps % parameters.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= parameters.BATCH_SIZE:

                    minibatch = self.buffer.sample()

                    _, _,critic_loss = self.sess.run([self.network.critic_train_op, self.network.actor_train_op,self.network.critic_loss],
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
                print('Episode %2i, initial heading: %7.3f, Reward: %7.3f, Final noise scale: %7.3f, critic loss: %7.3f' %
                      (ep, hdg0[0]*(1/TORAD), episode_reward, noise_scale,critic_loss))
            DISPLAYER.add_reward(episode_reward)
            # We save CNN weights every 500 epochs
            if ep % 500 == 0 and ep != 0:
                self.save("NetworkParam/"+ str(ep) +"_epochs")
        self.save("NetworkParam/"+"FinalParam")


    def playActor(self):
        self.load("NetworkParam/FinalParam")

        hdg0_rand_vec=[0,7,12]
        '''
        WIND CONDITIONS
        '''
        mean = 45 * TORAD
        std = 0.1 * TORAD
        wind_samples = 10
        w = wind(mean=mean, std=std, samples = wind_samples)

        try:
            for i in range(len(hdg0_rand_vec)):
                # Initial state
                WH = w.generateWind()
                hdg0_rand = hdg0_rand_vec[i]
                hdg0 = hdg0_rand * TORAD * np.ones(10)
                s = self.env.reset(hdg0,WH)
                episode_reward = 0
                episode_step=0
                v_episode=[]
                i_episode=[]
                while episode_step < 40: #not done:
                    if episode_step==0:
                        i_episode.append(hdg0_rand+WH[0]/TORAD-40)
                    else:
                        i_episode.append(s[0][-1]/TORAD)
                    s = np.reshape([s[0,:], s[1,:]], [self.state_size,1])

                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: s[None]})
                    a = np.clip(a, self.low_bound, self.high_bound)
                    s_, r   = self.env.act(a,WH)
                    episode_reward += r
                    v_episode.append(r)
                    episode_step += 1
                    s = s_
                DISPLAYER.displayVI(v_episode,i_episode,i)
                print("Episode reward :", episode_reward," for incidence: ",hdg0_rand)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")

    def playCritic(self):
        self.load("NetworkParam/FinalParam")

        hdg0_rand_vec=[0,7,12]
        '''
        WIND CONDITIONS
        '''
        mean = 45 * TORAD
        std = 0.1 * TORAD
        wind_samples = 10
        w = wind(mean=mean, std=std, samples = wind_samples)

        try:
            for i in range(len(hdg0_rand_vec)):
                # Initial state
                WH = w.generateWind()
                hdg0_rand = hdg0_rand_vec[i]
                hdg0 = hdg0_rand * TORAD * np.ones(10)
                s = self.env.reset(hdg0,WH)
                
                episode_reward = 0
                episode_step=0
                v_episode=[]
                i_episode=[]
                while episode_step < 30: #not done:
                    if episode_step==0:
                        i_episode.append(hdg0_rand+WH[0]/TORAD-40)
                    else:
                        i_episode.append(s[0][-1]/TORAD)
                    
                    # Critic policy
                    critic = [self.evaluate(s, -1.5),self.evaluate(s, -1.25),self.evaluate(s, -1),
                        self.evaluate(s, -0.75),self.evaluate(s, -0.5),self.evaluate(s, -0.25),self.evaluate(s, 0),self.evaluate(s, 0.25),
                            self.evaluate(s, 0.5),self.evaluate(s, 0.75),self.evaluate(s, 1),self.evaluate(s, 1.25),
                            self.evaluate(s, 1.5)]
                    a = np.argmax(critic)
                    if a == 0:
                        a = -1.5
                    if a == 1:
                        a = -1.25
                    if a == 2:
                        a = -1
                    if a == 3:
                        a = -0.75
                    if a == 4:
                        a = -0.5
                    if a == 5:
                        a = -0.25
                    if a == 6:
                        a = 0
                    if a == 7:
                        a = 0.25
                    if a == 8:
                        a = 0.5
                    if a == 9:
                        a = 0.75
                    if a == 10:
                        a = 1
                    if a == 11:
                        a = 1.25
                    if a == 12:
                        a = 1.5

                    s_, r   = self.env.act(a,WH)
                    episode_reward += r
                    v_episode.append(r)
                    episode_step += 1
                    s = s_
                DISPLAYER.displayVI(v_episode,i_episode,i+3)
                print("Episode reward :", episode_reward," for incidence: ",hdg0_rand)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            print("End of the demo")

    def save(self, name):
        """
        Save the weights of both of the networks into a .ckpt tensorflow session file
        :param name: Name of the file where the weights are saved
        """
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, name+".ckpt")
        print("Model saved in path: %s" % save_path)

    def load(self, name):
        """
        Load the weights of the 2 networks saved in the file into :ivar network
        :param name: name of the file containing the weights to load
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, name+".ckpt")

    def evaluate(self, state, action):
        """
        Evaluate the Q-value of a state-action pair  using the critic neural network.

        :param np.array state: state that we want to evaluate.
        :param float action: action that we want to evaluate (has to be between permitted bounds)
        :return: The continuous action value.
        """
        s = np.reshape([state[0, :], state[1, :]], (1,self.state_size, 1))
        a = np.reshape(action, (1,self.action_size, 1))
        q = self.sess.run(
            self.network.q_values_of_given_actions,
            feed_dict={
                self.network.state_ph: s,
                self.network.action_ph: a})
        return q
