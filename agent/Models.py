import keras as k
import tensorflow as tf
from tensorflow import keras
import numpy as np
from agent.Losses import Losses


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


# class DenseQnetwork():
#
#     def __init__(self, input_shape, h_size, n_actions, scope_var, device):
#         with tf.device(device):
#             with tf.variable_scope(scope_var):
#                 tf.set_random_seed(1)
#
#                 self.imageIn = tf.placeholder(shape=input_shape, dtype=tf.float32)
#                 self.dense1 = tf.layers.Dense(32, activation='elu')(self.imageIn)
#                 self.Qout = tf.layers.Dense(n_actions, activation='linear')(self.dense1)
#                 self.predict = tf.argmax(self.Qout, 1)
#
#                 # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q
#                 # values.
#                 self.targetQ = tf.placeholder(shape=[None, n_actions], dtype=tf.float32)
#                 self.actions = tf.placeholder(shape=[None, n_actions], dtype=tf.int32)
#                 self.td_error = tf.square(self.targetQ - self.Qout)
#                 self.loss = tf.reduce_mean(self.td_error)
#                 self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
#                 self.updateModel = self.trainer.minimize(self.loss)
#
#     def prediction(self, sess, s):
#
#         return sess.run(self.predict, feed_dict={self.imageIn: s})
#
#     def Qprediction(self, sess, s):
#
#         return sess.run(self.Qout, feed_dict={self.imageIn: s})
#
#     def train(self, sess, s, targetQ):
#         return sess.run([self.updateModel, self.loss],
#                         feed_dict={self.imageIn: s, self.targetQ: targetQ})

#class Linear():

    # from https://github.com/awjuliani/DeepRL-Agents

#    def __init__(self, input_shape, h_size, n_actions, scope_var, device):
#        self.g = tf.Graph()
#        with self.g.as_default():
#            with tf.device(device):
#                with tf.variable_scope(scope_var):
#                    tf.set_random_seed(1)
#
#                    self.imageIn = tf.placeholder(shape=input_shape, dtype=tf.float32)
#                    self.dense1 = tf.layers.Dense(8, activation='linear')(self.imageIn)
#                    self.Qout = tf.layers.Dense(n_actions, activation='linear')(self.imageIn)
#                    self.predict = tf.argmax(self.Qout, 1)

                    # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q
                    # values.

#                    self.ISWeights_ = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
#                    self.targetQ = tf.placeholder(shape=[None, n_actions], dtype=tf.float32)
#                    self.actions = tf.placeholder(shape=[None, n_actions], dtype=tf.int32)
#                    self.td_error = tf.square(self.targetQ - self.Qout)
#                    self.loss = tf.reduce_mean(self.ISWeights_ * self.td_error)
#                    self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
#                    self.updateModel = self.trainer.minimize(self.loss)
#
#            graph_init_op = tf.global_variables_initializer()
#        self.tf_sess = tf.Session(graph=self.g)
#        self.tf_sess.run(graph_init_op)

#    def prediction(self, sess, s):

#        return self.tf_sess.run(self.predict, feed_dict={self.imageIn: s})

#    def Qprediction(self, sess, s):

#        return self.tf_sess.run(self.Qout, feed_dict={self.imageIn: s})

#    def train(self, sess, s, targetQ, imp_w):

#        return self.tf_sess.run([self.updateModel, self.loss],
#                                feed_dict={self.imageIn: s, self.targetQ: targetQ, self.ISWeights_: imp_w})


class Linear():

    # from https://github.com/awjuliani/DeepRL-Agents

    def __init__(self, input_shape, h_size, n_actions, scope_var, device):

        self.model = k.models.Sequential()
        self.model.add(k.layers.Dense(8,input_shape=[1],kernel_initializer='normal',activation='linear'))
        self.model.add(k.layers.Dense(4, kernel_initializer='normal'))
        self.model.compile(loss='mean_squared_error', optimizer='adam')


    def prediction(self, sess, s):

        s=np.array(s)

        a = self.model.predict(s)

        return np.argmax(a,1)

    def Qprediction(self, sess, s):

        return self.model.predict(s)

    def train(self, sess, s, targetQ, imp_w):

        self.model.fit(s, targetQ, epochs=1, verbose=0)

        return [1,2]


class DenseModel(keras.Model):

    def __init__(self, n_actions, input_shape):
        super(DenseModel, self).__init__(name="DenseQnetwork")
        self.dense1 = keras.layers.Dense(8, activation='linear',)
        #self.Qout = keras.layers.Dense(n_actions, activation='linear')

        # Have the network estimate the Advantage function as an intermediate layer
        self.A = keras.layers.Dense(n_actions + 1, activation='linear')
        self.Qout = keras.layers.Lambda(lambda i: keras.backend.expand_dims(i[:, 0], -1) + i[:, 1:] - keras.backend.mean(i[:, 1:], keepdims=True),output_shape=(n_actions,))

    def call(self, x):
        x = self.dense1(x)
        x = self.A(x)
        x = self.Qout(x)
        return x


class DenseQnetwork():

    def __init__(self, input_shape, h_size, n_actions, scope_var, device):
        tf.set_random_seed(1)
        with tf.device(device):
            self.model = DenseModel(n_actions, input_shape)
            self.model.compile(loss=Losses.huber_loss_mean, optimizer=tf.train.AdamOptimizer(0.001))

    def prediction(self, sess, s):

        s = np.array(s, dtype=np.float32)

        a = self.model.predict(s)

        return np.argmax(a, 1)

    def Qprediction(self, sess, s):

        s = np.array(s, dtype=np.float32)

        return self.model.predict(s)

    def train(self, sess, s, targetQ, imp_w):

        self.model.fit(s, targetQ, epochs=1, verbose=0)

        return [None,None]




