
import tensorflow as tf
import numpy as np

import gym
from gym import wrappers
import random

from QNetwork import Network

from Environment import Environment
from ExperienceBuffer import ExperienceBuffer

from Displayer import DISPLAYER
import parameters


class Agent:

    def __init__(self, sess):
        print("Initializing the agent...")

        self.sess = sess
        self.env = Environment()
        self.state_size = self.env.get_state_size()[0]
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

        for ep in range(1, parameters.TRAINING_STEPS+1):

            episode_reward = 0
            episode_step = 0
            done = False

            # Initialize exploration noise process
            noise_process = np.zeros(self.action_size)
            noise_scale = (parameters.NOISE_SCALE_INIT *
                           parameters.NOISE_DECAY**ep) * \
                (self.high_bound - self.low_bound)

            # Initial state
            s = self.env.reset()
            render = (ep % parameters.RENDER_FREQ == 0 and parameters.DISPLAY)
            self.env.set_render(render)

            while episode_step < parameters.MAX_EPISODE_STEPS and not done:

                # choose action based on deterministic policy
                a, = self.sess.run(self.network.actions,
                                   feed_dict={self.network.state_ph: s[None]})

                # add temporally-correlated exploration noise to action
                # (using an Ornstein-Uhlenbeck process)
                noise_process = parameters.EXPLO_THETA * \
                    (parameters.EXPLO_MU - noise_process) + \
                    parameters.EXPLO_SIGMA * np.random.randn(self.action_size)

                a += noise_scale * noise_process

                s_, r, done, info = self.env.act(a)
                episode_reward += r

                self.buffer.add((s, a, r, s_, 0.0 if done else 1.0))

                # update network weights to fit a minibatch of experience
                if self.total_steps % parameters.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= parameters.BATCH_SIZE:

                    minibatch = self.buffer.sample()

                    _, _ = self.sess.run([self.network.critic_train_op, self.network.actor_train_op],
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
                print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                      (ep, episode_reward, episode_step, noise_scale))
            DISPLAYER.add_reward(episode_reward)
            # We save CNN weights every 1000 epochs
            if ep % 1000 == 0 and ep != 0:
                self.save("NetworkParam/"+ str(ep) +"_epochs")


    def play(self, number_run):
        self.load("NetworkParam/FinalParam")
        print("Playing for", number_run, "runs")

        self.env.set_render(True)
        try:
            for i in range(number_run):

                s = self.env.reset()
                episode_reward = 0
                done = False

                while not done:

                    a, = self.sess.run(self.network.actions,
                                       feed_dict={self.network.state_ph: s[None]})

                    s, r, done, info = self.env.act(a)
                    episode_reward += r
                
                print("Episode reward :", episode_reward)

        except KeyboardInterrupt as e:
            pass

        except Exception as e:
            print("Exception :", e)

        finally:
            self.env.set_render(False)
            print("End of the demo")
            self.env.close()

    def close(self):
        self.env.close()

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
