
import tensorflow as tf

def build_actor(states, bounds, action_size, trainable, scope, reuse):
    """
    Builds actor CNN in the current tensorflow model under given scope name

    :param states: batch of states for learning or prediction
    :param bounds: list of minimum and maximum values of continuous action
    :param action_size: size of continuous action
    :param trainable: boolean that permits to fit neural network in the associated scope if value is True
    :param scope: builds an actor network under a given scope name, to be reused under this name

    :return: actions chosen by the actor network for each state of the batch
    """
    with tf.variable_scope(scope):
        '''
        # Convolutional and Pooling Layers #1 for incidence
        conv1_i = tf.layers.conv1d(
          inputs=states[:,0:60,:],
          filters=32,
          kernel_size=5,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv1_i', reuse = reuse)
        pool1_i = tf.layers.max_pooling1d(inputs=conv1_i, pool_size=2, strides=2)

        # Convolutional and Pooling Layers #1 for velocity
        conv1_v = tf.layers.conv1d(
          inputs=states[:,60:120,:],
          filters=32,
          kernel_size=5,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv1_v', reuse = reuse)
        pool1_v = tf.layers.max_pooling1d(inputs=conv1_v, pool_size=2, strides=2)
        
        # Convolutional Layer #2 and Pooling Layer #2 for incidence
        conv2_i = tf.layers.conv1d(
          inputs=pool1_i,
          filters=70,
          kernel_size=5,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv2_i', reuse = reuse)
        pool2_i = tf.layers.max_pooling1d(inputs=conv2_i, pool_size=2, strides=2)

        # Convolutional Layer #2 and Pooling Layer #2 for velocity
        conv2_v = tf.layers.conv1d(
          inputs=pool1_v,
          filters=70,
          kernel_size=5,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv2_v', reuse = reuse)
        pool2_v = tf.layers.max_pooling1d(inputs=conv2_v, pool_size=2, strides=2)

        # Dense Layer for incidence
        pool2_i_flat = tf.reshape(pool2_i, [-1, 15 * 70])
        dense_i = tf.layers.dense(inputs=pool2_i_flat, units=256, activation=tf.nn.relu,trainable=trainable,name='dense_i', reuse = reuse)
        #dropout_i = tf.layers.dropout(
            #inputs=dense_i, rate=0.4, training=trainable, name = 'dropout_i')
        dense_i = tf.layers.batch_normalization(dense_i, trainable = trainable, name = 'bnormi')
        
        # Dense Layer for velocity
        pool2_v_flat = tf.reshape(pool2_v, [-1, 15 * 70])
        dense_v = tf.layers.dense(inputs=pool2_v_flat, units=256, activation=tf.nn.relu,trainable=trainable,name='dense_v', reuse = reuse)
        #dropout_v = tf.layers.dropout(
            #inputs=dense_v, rate=0.4, training=trainable, name = 'dropout_v')
        dense_v = tf.layers.batch_normalization(dense_v, trainable = trainable, name = 'bnormv')

        #Concatenation of both plus another dense
        merge = tf.concat([dense_v,dense_i],0)
        dense = tf.layers.dense(inputs=merge,units=1024, activation=tf.nn.relu,trainable=trainable,name='dense', reuse = reuse)
        dense = tf.layers.batch_normalization(dense, trainable = trainable, name = 'bnorm1')

        dense2 = tf.layers.dense(inputs=dense,units=256, activation=tf.nn.relu,trainable=trainable,name='dense2', reuse = reuse)
        dense2 = tf.layers.batch_normalization(dense2, trainable = trainable, name = 'bnorm2')

        dense3 = tf.layers.dense(inputs=dense2,units=64, activation=tf.nn.relu,trainable=trainable,name='dense3', reuse = reuse)
        dense3 = tf.layers.batch_normalization(dense3, trainable = trainable, name = 'bnorm3')
        print("penultima layer shape: ",dense3.shape)
        #Finql value that will be scaled between 0 and 1
        actions_unscaled = tf.layers.dense(dense3, action_size,
                                           trainable=trainable,
                                           name='dense_i_v_out', reuse = reuse)
        print("ultima layer shape: ",actions_unscaled.shape)
        # bound the actions to the valid range
        low_bound, high_bound = bounds
        valid_range = high_bound - low_bound
        actions = low_bound + tf.nn.sigmoid(actions_unscaled) * valid_range
        '''

        
        conv_i = tf.layers.conv1d(states[:,0:60,:], filters = 30,kernel_size = 5, strides = 1, padding = "same", trainable=trainable,
                                 activation=tf.nn.leaky_relu, name='conv_i', reuse = reuse)
        pool_i_1 = tf.layers.max_pooling1d(conv_i,pool_size = 2, strides = 2)

        conv_v = tf.layers.conv1d(states[:,60:120,:], filters=30,kernel_size = 5, strides = 1, padding = "same", trainable=trainable,
                                 activation=tf.nn.leaky_relu, name='conv_v', reuse = reuse)
        pool_v_1 = tf.layers.max_pooling1d(conv_v,pool_size = 2, strides = 2)

        conv_i_2 = tf.layers.conv1d(pool_i_1, filters=60, kernel_size=5,trainable=trainable, padding = "same",
                                    activation=tf.nn.leaky_relu, name='conv_i_2', reuse = reuse)
        conv_i_2 = tf.layers.max_pooling1d(conv_i_2,pool_size = 2, strides = 2)

        conv_v_2 = tf.layers.conv1d(pool_v_1, filters=60, kernel_size=5,trainable=trainable, padding = "same",
                                    activation=tf.nn.leaky_relu, name='conv_v_2', reuse = reuse)
        conv_v_2 = tf.layers.max_pooling1d(conv_v_2,pool_size = 2, strides = 2)

        flatv = tf.contrib.layers.flatten(conv_v_2,scope=scope)
        hidden_v = tf.layers.dense(flatv, 50, trainable=trainable,
                                   activation=tf.nn.leaky_relu, name='first_hidden_v', reuse = reuse)
        #hidden_v = tf.layers.batch_normalization(hidden_v, trainable = trainable, name = 'bnorm1')
        hidden_v = tf.layers.dropout(inputs=hidden_v, rate=0.4, training=trainable, name = 'dropout1')

        flati = tf.contrib.layers.flatten(conv_i_2,scope=scope)
        hidden_i = tf.layers.dense(flati, 50, trainable=trainable,
                                   activation=tf.nn.leaky_relu, name='first_hidden_i', reuse = reuse)
        hidden_i = tf.layers.batch_normalization(hidden_i, trainable = trainable, name = 'bnorm2')
        hidden_i = tf.layers.dropout(inputs=hidden_i, rate=0.4, training=trainable, name = 'dropout2')

        merge = tf.concat([hidden_i,hidden_v],axis = 1)
        

        hidden_i = tf.layers.dense(merge, 1024, trainable=trainable,
                                   activation=tf.nn.leaky_relu, name='dense1', reuse = reuse)
        #hidden_i = tf.layers.batch_normalization(hidden_i, trainable = trainable, name = 'bnorm3')
        hidden_i = tf.layers.dropout(inputs=hidden_i, rate=0.4, training=trainable, name = 'dropout3')

        hidden_v = tf.layers.dense(hidden_i, 128, trainable=trainable,
                                   activation=tf.nn.leaky_relu, name='dense2', reuse = reuse)
        #hidden_v = tf.layers.batch_normalization(hidden_v, trainable = trainable, name = 'bnorm4')
        hidden_v = tf.layers.dropout(inputs=hidden_v, rate=0.4, training=trainable, name = 'dropout4')

        hidden_i_2 = tf.layers.dense(hidden_v, 32, trainable=trainable,
                                   activation=tf.nn.leaky_relu, name='dense3', reuse = reuse)
        #hidden_i_2 = tf.layers.batch_normalization(hidden_i_2, trainable = trainable, name = 'bnorm5')
        hidden_i_2 = tf.layers.dropout(inputs=hidden_i_2, rate=0.4, training=trainable, name = 'dropout5')
        
        

        # conv_i_v = tf.layers.conv1d(states, filters = 40,kernel_size = 50, strides = 1, padding = "same", trainable=trainable,
        #                          activation=tf.nn.relu, name='conv_i_v', reuse = reuse)
        # conv_i_v_2 = tf.layers.conv1d(conv_i_v, filters=40, kernel_size=50, strides=1, padding="same", trainable=trainable,
        #                             activation=tf.nn.relu, name='conv_i_v_2', reuse=reuse)
        # flat = tf.contrib.layers.flatten(conv_i_v, scope = scope)

        #hidden_i_v = tf.layers.dense(hidden_v_2, 80, trainable=trainable,
                                   #activation=tf.nn.relu, name='dense_i_v', reuse = reuse)

        # hidden_i_v = tf.layers.batch_normalization(hidden_i_v, trainable = trainable, name = 'bnorm1')

        #hidden_i_v_2 = tf.layers.dense(hidden_i_v, 40, trainable=trainable,
                                   #activation=tf.nn.relu, name='dense_i_v_2', reuse = reuse)

        # hidden_i_v_2 = tf.layers.batch_normalization(hidden_i_v_2, trainable = trainable, name = 'bnorm2')

        
        actions_unscaled = tf.layers.dense(hidden_i_2, action_size,
                                           trainable=trainable, name='dense_i_v_out', reuse = reuse)
        
        # bound the actions to the valid range
        low_bound, high_bound = bounds
        valid_range = high_bound - low_bound
        actions = low_bound + tf.nn.sigmoid(actions_unscaled) * valid_range
        
        
    return actions


def build_critic(states, actions, trainable, reuse, scope):
    """
    Builds actor CNN in the current tensorflow model under given scope name

    :param states: batch of states for learning or prediction of Q-value
    :param action: batch of actions for learning or prediction of Q-value
    :param trainable: boolean that permits to fit neural network in the associated scope if value is True
    :param reuse: boolean that determines if the networks has to be built or reused when build_critic function is called
    :param scope: builds an actor network under a given scope name, to be reused under this name

    :return: q_values for each state-action pair of the given batch
    """
    with tf.variable_scope(scope):

        # states_actions = tf.concat([states, actions], axis=1)

        # Convolutional and Pooling Layers #1 for incidence
        conv1_i = tf.layers.conv1d(
          inputs=states[:,0:60,:],
          filters=32,
          kernel_size=30,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv1_i', reuse = reuse)
        pool1_i = tf.layers.max_pooling1d(inputs=conv1_i, pool_size=2, strides=2)

        # Convolutional and Pooling Layers #1 for velocity
        conv1_v = tf.layers.conv1d(
          inputs=states[:,60:120,:],
          filters=32,
          kernel_size=30,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv1_v', reuse = reuse)
        pool1_v = tf.layers.max_pooling1d(inputs=conv1_v, pool_size=2, strides=2)
        
        # Convolutional Layer #2 and Pooling Layer #2 for incidence
        conv2_i = tf.layers.conv1d(
          inputs=pool1_i,
          filters=64,
          kernel_size=30,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv2_i', reuse = reuse)
        pool2_i = tf.layers.max_pooling1d(inputs=conv2_i, pool_size=2, strides=2)

        # Convolutional Layer #2 and Pooling Layer #2 for velocity
        conv2_v = tf.layers.conv1d(
          inputs=pool1_v,
          filters=64,
          kernel_size=30,
          padding="same",
          strides = 1, trainable=trainable, activation=tf.nn.relu, name='conv2_v', reuse = reuse)
        pool2_v = tf.layers.max_pooling1d(inputs=conv2_v, pool_size=2, strides=2)

        merge = tf.concat([pool2_i, pool2_v], axis=1)
        merge_flat = tf.contrib.layers.flatten(merge, scope=scope)

        # Dense Layer 1
        
        dense1 = tf.layers.dense(inputs=merge_flat, units=256, activation=tf.nn.relu,trainable=trainable,name='dense_1', reuse = reuse)
        dropout_i = tf.layers.dropout(
            inputs=dense1, rate=0.4, training=trainable, name = 'dropout_1')
        
        # Dense Layer for velocity
        
        dense2 = tf.layers.dense(inputs=dense1, units=128, activation=tf.nn.relu,trainable=trainable,name='dense_2', reuse = reuse)
        dropout_v = tf.layers.dropout(
            inputs=dense2, rate=0.4, training=trainable, name = 'dropout_2')


        merge_with_action = tf.concat([tf.expand_dims(dense2,axis=2),actions],axis = 1)
        merge_with_action_flat = tf.contrib.layers.flatten(merge_with_action)


        hidden_i_v_2 = tf.layers.dense(merge_with_action_flat, 40, trainable=trainable,reuse=reuse,
                                       activation=tf.nn.relu, name='dense_i_v_a')
        hidden_i_v_3 = tf.layers.dense(hidden_i_v_2, 20, trainable=trainable, reuse=reuse,
                                      activation=tf.nn.relu, name='dense_i_v_a_2')
        q_values = tf.layers.dense(hidden_i_v_3, 1,
                                   trainable=trainable, reuse = reuse, name='dense_i_v_a_out')
      

        '''
        conv_i = tf.layers.conv1d(states[:, 0:60, :], reuse = reuse, filters=40, kernel_size=50, strides=1, padding="same",
                                  trainable=trainable,
                                  activation=tf.nn.relu, name='conv_i')
        conv_v = tf.layers.conv1d(states[:, 60:120, :], filters=40, kernel_size=50, strides=1, padding="same",
                                  trainable=trainable, reuse = reuse,
                                  activation=tf.nn.relu, name='conv_v')
        conv_i_2 = tf.layers.conv1d(conv_i, filters=30, kernel_size=20,
                                    trainable=trainable, padding="same", reuse = reuse,
                                    activation=tf.nn.relu, name='conv_i_2')
        conv_i_2 = tf.layers.max_pooling1d(conv_i_2, pool_size=2, strides = 1)

        conv_v_2 = tf.layers.conv1d(conv_v, filters=20, kernel_size=20,
                                    trainable=trainable, padding="same", reuse= reuse,
                                    activation=tf.nn.relu, name='conv_v_2')
        conv_v_2 = tf.layers.max_pooling1d(conv_v_2, pool_size=2, strides = 1)
        hidden_i = tf.layers.dense(conv_i_2, 120, trainable=trainable, reuse = reuse,
                                   activation=tf.nn.relu, name='dense_i')
        hidden_v = tf.layers.dense(conv_v_2, 120, trainable=trainable, reuse = reuse,
                                   activation=tf.nn.relu, name='dense_v')
        hidden_i_2 = tf.layers.dense(hidden_i, 60, trainable=trainable, reuse = reuse,
                                     activation=tf.nn.relu, name='dense_i_2')
        hidden_v_2 = tf.layers.dense(hidden_v, 60, trainable=trainable, reuse = reuse,
                                     activation=tf.nn.relu, name='dense_v_2')
        merge = tf.concat([hidden_i_2, hidden_v_2], axis=1)
        merge_flat = tf.contrib.layers.flatten(merge, scope=scope)
        print("merge flat shape = ", merge_flat.shape)
        # conv_i_v = tf.layers.conv1d(states, filters=40, kernel_size=50, strides=1, padding="same", trainable=trainable,
        #                             activation=tf.nn.relu, name='conv_i_v', reuse=reuse)
        # conv_i_v_2 = tf.layers.conv1d(conv_i_v, filters=40, kernel_size=50, strides=1, padding="same",
        #                               trainable=trainable,
        #                               activation=tf.nn.relu, name='conv_i_v_2', reuse=reuse)
        # flat = tf.contrib.layers.flatten(conv_i_v_2, scope=scope)

        # Ca se separe ici entre les deux methodes

        hidden_i_v = tf.layers.dense(merge_flat, 80, trainable=trainable,
                                     activation=tf.nn.relu, name='dense_i_v', reuse=reuse)
        # hidden_a = tf.layers.dense(actions, 1, trainable=trainable,reuse=reuse,
                                  #   activation=tf.nn.relu, name='dense_a')
        print("actions: ",actions)
        merge_with_action = tf.concat([tf.expand_dims(hidden_i_v,axis=2),actions],axis = 0)
        merge_with_action_flat = tf.contrib.layers.flatten(merge_with_action)

        hidden_i_v_2 = tf.layers.dense(merge_with_action_flat, 40, trainable=trainable,reuse=reuse,
                                       activation=tf.nn.relu, name='dense_i_v_a')
        hidden_i_v_3 = tf.layers.dense(hidden_i_v_2, 20, trainable=trainable, reuse=reuse,
                                      activation=tf.nn.relu, name='dense_i_v_a_2')
        q_values = tf.layers.dense(hidden_i_v_3, 1,
                                   trainable=trainable, reuse = reuse, name='dense_i_v_a_out')
        '''
    return q_values


def get_vars(scope, trainable):
    if trainable:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def copy_vars(src_vars, dest_vars, tau, name):
    update_dest = []
    for src_var, dest_var in zip(src_vars, dest_vars):
        op = dest_var.assign(tau * src_var + (1 - tau) * dest_var)
        update_dest.append(op)
    return tf.group(*update_dest, name=name)


def l2_regularization(vars):
    reg = 0
    for var in vars:
        if not 'bias' in var.name:
            reg += 1e-6 * 0.5 * tf.nn.l2_loss(var)
    return reg
