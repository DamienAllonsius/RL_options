import tensorflow as tf


class DualQnetwork():

    # from https://github.com/awjuliani/DeepRL-Agents

    def __init__(self, input_shape, h_size, n_actions, scope_var, device):
        with tf.device(device):
            with tf.variable_scope(scope_var):
                tf.set_random_seed(1)

                self.imageIn = tf.placeholder(shape=input_shape, dtype=tf.float32)
                self.conv1 = tf.layers.Conv2D(16, 3, (3, 3), padding='VALID', activation='elu')(self.imageIn)
                self.conv2 = tf.layers.Conv2D(32, 3, (3, 3), padding='VALID', activation='elu')(self.conv1)
                self.conv3 = tf.layers.Conv2D(32, 3, (1, 1), padding='VALID', activation='elu')(self.conv2)
                self.conv4 = tf.layers.Conv2D(h_size, 2, (1, 1), padding='VALID', activation='elu')(self.conv3)

                # We take the output from the final convolutional layer and split it into separate advantage and value
                # streams.
                self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
                self.streamA = tf.layers.flatten(self.streamAC)
                self.streamV = tf.layers.flatten(self.streamVC)
                xavier_init = tf.contrib.layers.xavier_initializer()
                self.AW = tf.Variable(xavier_init([h_size // 2, n_actions]))
                self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
                self.Advantage = tf.matmul(self.streamA, self.AW)
                self.Value = tf.matmul(self.streamV, self.VW)

                # Then combine them together to get our final Q-values.
                self.Qout = self.Value + tf.subtract(self.Advantage,
                                                     tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
                self.predict = tf.argmax(self.Qout, 1)

                # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q
                # values.

                self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
                self.targetQ = tf.placeholder(shape=[None, n_actions], dtype=tf.float32)
                self.actions = tf.placeholder(shape=[None, n_actions], dtype=tf.int32)
                self.td_error = tf.square(self.targetQ - self.Qout)
                self.loss = tf.reduce_mean(self.ISWeights_ * self.td_error)
                self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
                self.updateModel = self.trainer.minimize(self.loss)

    def prediction(self, sess, s):

        return sess.run(self.predict, feed_dict={self.imageIn: s})

    def Qprediction(self, sess, s):

        return sess.run(self.Qout, feed_dict={self.imageIn: s})

    def train(self, sess, s, targetQ, imp_w):
        return sess.run([self.updateModel, self.loss],
                        feed_dict={self.imageIn: s, self.targetQ: targetQ, self.ISWeights_: imp_w})


class DenseQnetwork():

    def __init__(self, input_shape, h_size, n_actions, scope_var, device):
        with tf.device(device):
            with tf.variable_scope(scope_var):
                tf.set_random_seed(1)

                self.imageIn = tf.placeholder(shape=input_shape, dtype=tf.float32)
                self.dense1 = tf.layers.Dense(32, activation='elu')(self.imageIn)
                self.Qout = tf.layers.Dense(n_actions, activation='linear')(self.dense1)
                self.predict = tf.argmax(self.Qout, 1)

                # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q
                # values.
                self.targetQ = tf.placeholder(shape=[None, n_actions], dtype=tf.float32)
                self.actions = tf.placeholder(shape=[None, n_actions], dtype=tf.int32)
                self.td_error = tf.square(self.targetQ - self.Qout)
                self.loss = tf.reduce_mean(self.td_error)
                self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
                self.updateModel = self.trainer.minimize(self.loss)

    def prediction(self, sess, s):

        return sess.run(self.predict, feed_dict={self.imageIn: s})

    def Qprediction(self, sess, s):

        return sess.run(self.Qout, feed_dict={self.imageIn: s})

    def train(self, sess, s, targetQ):
        return sess.run([self.updateModel, self.loss],
                        feed_dict={self.imageIn: s, self.targetQ: targetQ})




