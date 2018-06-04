
import tensorflow as tf
import numpy as np

import parameters


class Network:

    def __init__(self, state_size, action_size, low_bound, high_bound):

        self.state_size = state_size
        self.action_size = action_size
        self.low_bound = low_bound
        self.high_bound = high_bound

        # placeholders
        self.state_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.state_size,1])
        self.action_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.action_size,1])
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.next_state_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.state_size,1])
        self.is_not_terminal_ph = tf.placeholder(
            dtype=tf.float32, shape=[None])

        # Main actor network
        with tf.variable_scope('actor'):
            self.actions = self.generate_actor_network(self.state_ph,
                                                       trainable=True,
                                                       reuse=False)

        # Target actor network
        with tf.variable_scope('slow_target_actor', reuse=False):
            self.slow_target_next_actions = tf.stop_gradient(
                self.generate_actor_network(self.next_state_ph,
                                            trainable=False,
                                            reuse=False))

        with tf.variable_scope('critic') as scope:
            # Critic applied to state_ph and a given action (to train critic)
            self.q_values_of_given_actions = self.generate_critic_network(
                self.state_ph, self.action_ph, trainable=True, reuse=False)
            # Critic applied to state_ph and the current policy's outputted
            # actions for state_ph (to train actor)
            self.actions = tf.expand_dims(self.actions, axis=2)
            self.q_values_of_suggested_actions = self.generate_critic_network(
                self.state_ph, self.actions, trainable=True, reuse=True)

        # slow target critic network
        with tf.variable_scope('slow_target_critic', reuse=False):
            # Slow target critic applied to slow target actor's outputted
            # actions for next_state_ph (to train critic)
            self.slow_target_next_actions = tf.expand_dims(self.slow_target_next_actions, axis=2)
            self.slow_q_values_next = tf.stop_gradient(
                self.generate_critic_network(self.next_state_ph,
                                             self.slow_target_next_actions,
                                             trainable=False, reuse=False))

        # isolate vars for each network
        self.actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        self.slow_target_actor_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
        self.critic_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        self.slow_target_critic_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

        # update values for slowly-changing targets towards current actor and
        # critic
        update_slow_target_ops = []
        for i, slow_target_actor_var in enumerate(self.slow_target_actor_vars):
            update_slow_target_actor_op = slow_target_actor_var.assign(
                parameters.UPDATE_TARGET_RATE * self.actor_vars[i] +
                (1 - parameters.UPDATE_TARGET_RATE) * slow_target_actor_var)
            update_slow_target_ops.append(update_slow_target_actor_op)

        for i, slow_target_var in enumerate(self.slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(
                parameters.UPDATE_TARGET_RATE * self.critic_vars[i] +
                (1 - parameters.UPDATE_TARGET_RATE) * slow_target_var)
            update_slow_target_ops.append(update_slow_target_critic_op)

        self.update_slow_targets_op = tf.group(*update_slow_target_ops,
                                               name='update_slow_targets')

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + parameters.DISCOUNT*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(self.reward_ph, 1) + \
            tf.expand_dims(self.is_not_terminal_ph, 1) * parameters.DISCOUNT * \
            self.slow_q_values_next

        # 1-step temporal difference errors
        td_errors = targets - self.q_values_of_given_actions

        # Critic loss and optimization
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        self.critic_loss = critic_loss ####linea nueva
        for var in self.critic_vars:
            if not 'bias' in var.name:
                critic_loss += 1e-6 * 0.5 * tf.nn.l2_loss(var)
                self.critic_loss = critic_loss ####linea nueva

        critic_trainer = tf.train.AdamOptimizer(
            parameters.CRITIC_LEARNING_RATE)
        self.critic_train_op = critic_trainer.minimize(critic_loss)

        # Actor loss and optimization
        actor_loss = -1 * tf.reduce_mean(self.q_values_of_suggested_actions)
        self.actor_loss = actor_loss ####linea nueva
        for var in self.actor_vars:
            if not 'bias' in var.name:
                actor_loss += 1e-6 * 0.5 * tf.nn.l2_loss(var)
                self.actor_loss = actor_loss ####linea nueva

        actor_trainer = tf.train.AdamOptimizer(parameters.ACTOR_LEARNING_RATE)

        self.actor_train_op = actor_trainer.minimize(actor_loss,
                                                     var_list=self.actor_vars)

    # Actor definition :
    def generate_actor_network(self, states, trainable, reuse):
        conv1 = tf.layers.conv1d( states, filters = 50,kernel_size = 50, strides = 5, padding = "same", trainable=trainable,
                                 activation=tf.nn.leaky_relu, name='conv1', reuse = reuse)
        pool1 = tf.layers.max_pooling1d(conv1,pool_size = 2, strides = 2)

        conv2 = tf.layers.conv1d(pool1, filters=60, kernel_size=20, strides = 2,trainable=trainable, padding = "same",
                                    activation=tf.nn.leaky_relu, name='conv2', reuse = reuse)
        pool2 = tf.layers.max_pooling1d(conv2,pool_size = 2, strides = 2)

        conv3 = tf.layers.conv1d(pool2, filters=60, kernel_size=10, strides = 1,trainable=trainable, padding = "same",
                                    activation=tf.nn.leaky_relu, name='conv3', reuse = reuse)
        pool3 = tf.layers.max_pooling1d(conv3,pool_size = 2, strides = 2)

        flat = tf.contrib.layers.flatten(pool3)

        hidden = tf.layers.dense(flat, 128,
                                 trainable=trainable, reuse=reuse,
                                 activation=tf.nn.leaky_relu, name='dense')
        hidden_2 = tf.layers.dense(hidden, 16,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.leaky_relu, name='dense_1')
        hidden_3 = tf.layers.dense(hidden_2, 8,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.leaky_relu, name='dense_2')
        actions_unscaled = tf.layers.dense(hidden_3, self.action_size,
                                           trainable=trainable, reuse=reuse,
                                           name='dense_3')
        # bound the actions to the valid range
        valid_range = self.high_bound - self.low_bound
        actions = self.low_bound + \
            tf.nn.sigmoid(actions_unscaled) * valid_range
        return actions

    # Critic definition :
    def generate_critic_network(self, states, actions, trainable, reuse):
        state_action = tf.concat([states, actions], axis=1)

        conv1 = tf.layers.conv1d(state_action, filters = 50,kernel_size = 50, strides = 5, padding = "same", trainable=trainable,
                                 activation=tf.nn.leaky_relu, name='conv1', reuse = reuse)
        pool1 = tf.layers.max_pooling1d(conv1,pool_size = 2, strides = 2)

        conv2 = tf.layers.conv1d(pool1, filters=60, kernel_size=20, strides = 2,trainable=trainable, padding = "same",
                                    activation=tf.nn.leaky_relu, name='conv2', reuse = reuse)
        pool2 = tf.layers.max_pooling1d(conv2,pool_size = 2, strides = 2)

        conv3 = tf.layers.conv1d(pool2, filters=60, kernel_size=10, strides = 1,trainable=trainable, padding = "same",
                                    activation=tf.nn.leaky_relu, name='conv3', reuse = reuse)
        pool3 = tf.layers.max_pooling1d(conv3,pool_size = 2, strides = 2)

        flat = tf.contrib.layers.flatten(pool3)

        hidden = tf.layers.dense(flat, 32,
                                 trainable=trainable, reuse=reuse,
                                 activation=tf.nn.leaky_relu, name='dense')
        hidden2 = tf.layers.batch_normalization(hidden)
        hidden_2 = tf.layers.dense(hidden2, 16,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.leaky_relu, name='dense_1')
        hidden_3 = tf.layers.dense(hidden_2, 8,
                                   trainable=trainable, reuse=reuse,
                                   activation=tf.nn.leaky_relu, name='dense_2')
        q_values = tf.layers.dense(hidden_3, 1,
                                   trainable=trainable, reuse=reuse,
                                   name='dense_3')
        return q_values
