
import tensorflow as tf

from DDPG.Model import *
import DDPG.settings as settings


class Network:

    def __init__(self, state_size, action_size, low_bound, high_bound):
        self.actor_loss = 0
        self.critic_loss = 0
        self.state_size = 2*state_size # incidence and speed are concatenated in a line
        self.action_size = action_size
        self.bounds = (low_bound, high_bound)

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_size,1])
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.action_size,1])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None,self.state_size,1])
        self.is_not_done_ph = tf.placeholder(dtype=tf.float32, shape=[None])

        # Main actor network
        self.actions = build_actor(self.state_ph, self.bounds, self.action_size,
                                   trainable=True, scope='actor')

        # Main critic network
        self.q_values_of_given_actions = build_critic(
            self.state_ph, self.action_ph, trainable=True, reuse=False, scope='critic')
        self.actions = tf.expand_dims(self.actions, axis = 2)   # We need
        self.q_values_of_suggested_actions = build_critic(
            self.state_ph, self.actions, trainable=True, reuse=True, scope='critic')
        
        # Target actor network
        self.target_next_actions = tf.stop_gradient(
            build_actor(self.next_state_ph, self.bounds, self.action_size,
                        trainable=False, scope='target_actor'))

        # Target critic network
        self.target_next_actions = tf.expand_dims(self.target_next_actions,axis = 2)
        self.q_values_next = tf.stop_gradient(
            build_critic(self.next_state_ph, self.target_next_actions,
                         trainable=False, reuse=False, scope='target_critic'))

        # Isolate vars for each network
        self.actor_vars = get_vars('actor', trainable=True)
        self.critic_vars = get_vars('critic', trainable=True)
        self.vars = self.actor_vars + self.critic_vars

        self.target_actor_vars = get_vars('target_actor', trainable=False)
        self.target_critic_vars = get_vars('target_critic', trainable=False)
        self.target_vars = self.target_actor_vars + self.target_critic_vars

        # Initialize target critic vars to critic vars
        self.target_init = copy_vars(self.vars,
                                     self.target_vars,
                                     1, 'init_target')

        # Update values for target vars towards current actor and critic vars
        self.update_targets = copy_vars(self.vars,
                                        self.target_vars,
                                        settings.UPDATE_TARGET_RATE,
                                        'update_targets')

        # Compute the target value
        reward = tf.expand_dims(self.reward_ph, 1)
        not_done = tf.expand_dims(self.is_not_done_ph, 1)
        targets = reward + not_done * settings.DISCOUNT * self.q_values_next

        # 1-step temporal difference errors
        td_errors = targets - self.q_values_of_given_actions

        # Critic loss and optimization
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        critic_loss += l2_regularization(self.critic_vars)
        critic_trainer = tf.train.AdamOptimizer(settings.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)
        self.critic_loss = critic_loss

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        actor_loss += l2_regularization(self.actor_vars)
        actor_trainer = tf.train.AdamOptimizer(settings.ACTOR_LEARNING_RATE)
        self.actor_train_op = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)
        self.actor_loss=actor_loss

        # For test
        self.prediction = tf.stop_gradient(build_critic(self.state_ph, self.action_ph,
                                                       trainable=False, reuse=True,scope='critic'))