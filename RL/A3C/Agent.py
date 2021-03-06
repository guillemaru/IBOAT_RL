
from time import time

import tensorflow as tf
import numpy as np
import scipy.signal
import random
import math


from Environment import Environment
from Network import Network
from Displayer import DISPLAYER
from Saver import SAVER

import settings

import sys
sys.path.append("../../sim/")
from Simulator import TORAD
from mdp import ContinuousMDP
from environment import wind



# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1],
                                [1, -gamma],
                                x[::-1],
                                axis=0)[::-1]


# Copies one set of variables to another.
# Used to set worker network settings to those of global network.
def update_target_graph(from_scope, to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Agent:

    def __init__(self, worker_index, sess, render=False, master=False):

        self.worker_index = worker_index
        if master:
            self.name = 'global'
        else:
            print("Initialization of the agent", str(worker_index))
            self.name = 'Worker_' + str(worker_index)

        self.env = Environment()
        self.state_size = self.env.get_state_size()
        self.action_size = self.env.get_action_size()
        self.low_bound,self.high_bound = self.env.get_bounds()

        self.network = Network(self.state_size, self.action_size, self.name)
        self.update_local_vars = update_target_graph('global', self.name)

        self.starting_time = 0
        self.epsilon = settings.EPSILON_START

        if self.name != 'global':
            self.summary_writer = tf.summary.FileWriter("results/" + self.name,
                                                        sess.graph)

    def save(self, episode_step):
        # Save model
        SAVER.save(episode_step)

        # Save summary statistics
        summary = tf.Summary()
        summary.value.add(tag='Perf/Reward',
                          simple_value=np.mean(self.rewards_plus))
        summary.value.add(tag='Perf/Value',
                          simple_value=np.mean(self.next_values))
        summary.value.add(tag='Losses/Value',
                          simple_value=self.value_loss)
        summary.value.add(tag='Losses/Policy',
                          simple_value=self.policy_loss)
        summary.value.add(tag='Losses/Entropy',
                          simple_value=self.entropy)
        summary.value.add(tag='Losses/Grad Norm',
                          simple_value=self.grad_norm)
        self.summary_writer.add_summary(summary, self.nb_ep)
        self.summary_writer.flush()

    def train(self, sess, bootstrap_value):

        # Add the bootstrap value to our experience
        self.rewards_plus = np.asarray(self.rewards_buffer + [bootstrap_value])
        discounted_reward = discount(
            self.rewards_plus, settings.DISCOUNT)[:-1]

        self.next_values = np.asarray(self.values_buffer[1:] +
                                      [bootstrap_value])
        advantages = self.rewards_buffer + \
            settings.DISCOUNT * self.next_values - \
            self.values_buffer
        advantages = discount(
            advantages, settings.GENERALIZED_LAMBDA * settings.DISCOUNT)


        # Update the global network
        feed_dict = {
            self.network.discounted_reward: discounted_reward,
            self.network.inputs: self.states_buffer,
            self.network.actions: self.actions_buffer,
            self.network.advantages: advantages}
        losses = sess.run([self.network.value_loss,
                           self.network.policy_loss,
                           self.network.entropy,
                           self.network.grad_norm,
                           self.network.apply_grads],
                          feed_dict=feed_dict)

        # Get the losses for tensorboard
        self.value_loss, self.policy_loss, self.entropy = losses[:3]
        self.grad_norm, _ = losses[3:]


        # Reinitialize buffers and variables
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.values_buffer = []

    def work(self, sess, coord):
        print("Running", self.name, end='\n\n')
        self.starting_time = time()
        self.nb_ep = 1
        nearlyDone = 0
        with sess.as_default(), sess.graph.as_default():

            with coord.stop_on_exception():
                while not coord.should_stop():

                    self.states_buffer = []
                    self.actions_buffer = []
                    self.rewards_buffer = []
                    self.values_buffer = []
                    self.mean_values_buffer = []

                    self.total_steps = 0
                    episode_reward = 0
                    episode_step = 0

                    # Reset the local network to the global
                    sess.run(self.update_local_vars)

                    mean = 45 * TORAD
                    std = 0 * TORAD
                    wind_samples = 10
                    w = wind(mean=mean, std=std, samples = wind_samples)
                    WH = w.generateWind()
                    hdg0_rand = random.uniform(5,12) 
                    hdg0 = hdg0_rand * TORAD * np.ones(10)
                    s = self.env.reset(hdg0,WH)
                    
                    done = False
                    #if self.worker_index == 1 and render and settings.DISPLAY:
                    #    self.env.set_render(True)

                    #self.lstm_state = self.network.lstm_state_init
                    #self.initial_lstm_state = self.lstm_state

                    while not coord.should_stop() and not done and \
                            episode_step < settings.MAX_EPISODE_STEP:

                        
                        
                        
                        WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
                        s = np.reshape([s[0,:], s[1,:]], [2*self.state_size,1])

                        # Prediction of the policy and the value
                        feed_dict = {self.network.inputs: [s]}
                        policy, value = sess.run(
                            [self.network.policy,
                             self.network.value], feed_dict=feed_dict)

                        policy, value = policy[0], value[0][0]


                        if random.random() < self.epsilon:
                            action = random.choice([1.5,0,-1.5])

                        else:
                            # Choose an action according to the policy
                            action = np.random.choice([1.5,0,-1.5],
                                                      p=policy)

                        s_, v = self.env.act(action,WH)
                        
                        #reward  assignation algorithm
                        if episode_step==1:
                            r=0
                        elif s[int(self.state_size/2-2)]>(13*TORAD) and s[int(self.state_size/2-2)]<(15*TORAD) and v>0.63 and v<0.67 and action<0:
                            r=0.5
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

                        
                        #s_ = np.reshape(s_, [2*self.state_size,1])

                        # Store the experience
                        self.states_buffer.append(s)
                        self.actions_buffer.append(action) 
                        self.rewards_buffer.append(r)
                        self.values_buffer.append(value)
                        self.mean_values_buffer.append(value)
                        episode_reward += r
                        s = s_

                        episode_step += 1
                        self.total_steps += 1

                        # If we have more than MAX_LEN_BUFFER experiences, we
                        # apply the gradients and update the global network,
                        # then we empty the episode buffers
                        if len(self.states_buffer) == settings.MAX_LEN_BUFFER \
                                and not done:
     
                            feed_dict = {self.network.inputs: [np.reshape([s[0,:], s[1,:]], [2*self.state_size,1])]}
                            bootstrap_value = sess.run(
                                self.network.value,
                                feed_dict=feed_dict)

                            self.train(sess, bootstrap_value) #with this we change global network
                            sess.run(self.update_local_vars)
                            #self.initial_lstm_state = self.lstm_state

                    if len(self.states_buffer) != 0:
                        if done:
                            bootstrap_value = 0
                        else:
                            feed_dict = {self.network.inputs: [np.reshape([s[0,:], s[1,:]], [2*self.state_size,1])]}
                            bootstrap_value = sess.run(
                                self.network.value,
                                feed_dict=feed_dict)
                        self.train(sess, bootstrap_value)

                    if self.epsilon > settings.EPSILON_STOP:
                        self.epsilon -= settings.EPSILON_DECAY

                    self.nb_ep += 1

                    if not coord.should_stop():
                        DISPLAYER.add_reward(episode_reward, self.worker_index)

                    if (self.worker_index == 1 and
                            self.nb_ep % settings.DISP_EP_REWARD_FREQ == 0):
                        print('Episode %2i, Initial hdg: %2i, Reward: %7.3f, Steps: %i, '
                              'Epsilon: %7.3f' %
                              (self.nb_ep, hdg0_rand, episode_reward, episode_step,
                               self.epsilon))
                        print("Policy: ",policy)
                    if (self.worker_index == 1 and
                            self.nb_ep % settings.SAVE_FREQ == 0):
                        self.save(self.total_steps)

                    if time() - self.starting_time > settings.LIMIT_RUN_TIME:
                        coord.request_stop()


            self.summary_writer.close()

    def play(self, sess, number_run, path=''):
        print("Playing", self.name, "for", number_run, "runs")

        with sess.as_default(), sess.graph.as_default():
            hdg0_rand_vec=[0,7,13]
            '''
            WIND CONDITIONS
            '''
            mean = 45 * TORAD
            std = 0 * TORAD
            wind_samples = 10
            w = wind(mean=mean, std=std, samples = wind_samples)
            
            try:
                for i in range(number_run):

                    # Reset the local network to the global
                    if self.name != 'global':
                        sess.run(self.update_local_vars)

                    
                    WH = w.generateWind()
                    hdg0_rand = hdg0_rand_vec[i]
                    hdg0 = hdg0_rand * TORAD * np.ones(10)
                    s = self.env.reset(hdg0,WH)
                    episode_reward = 0
                    episode_step=0
                    v_episode=[]
                    i_episode=[]
                    done = False
                    
                    #self.lstm_state = self.network.lstm_state_init

                    while (not done and episode_step<70):
                        i_episode.append(round(s[0][-1]/TORAD))
                        s = np.reshape([s[0,:], s[1,:]], [2*self.state_size,1])
                        # Prediction of the policy
                        feed_dict = {self.network.inputs: [s]}
                        policy,value = sess.run(
                            [self.network.policy,
                             self.network.value], feed_dict=feed_dict)

                        policy = policy[0]

                        # Choose an action according to the policy
                        action = np.random.choice([1.5,0,-1.5], p=policy)
                        s_, r = self.env.act(action, WH)
                        if episode_step>12:
                            if np.mean(v_episode[-4:])>0.8:
                                #done=True
                                print("Done!")
                            else:
                                done = False
                        episode_reward += r
                        v_episode.append(r)
                        episode_step += 1
                        s=s_
                    DISPLAYER.displayVI(v_episode,i_episode,i)

                    print("Episode reward :", episode_reward)


            except KeyboardInterrupt as e:
                pass

            finally:
                print("End of the demo")

