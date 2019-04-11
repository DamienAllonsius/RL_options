import keras as k
import tensorflow as tf
from tensorflow import keras
import numpy as np
from agent.Losses import Losses


class ConvolutionalLayersShared:

    sharedConv_index = 0

    def __init__(self):
        self.conv1 = keras.layers.Conv2D(16, 3, (3, 3), padding='VALID', activation='elu')
        self.conv2 = keras.layers.Conv2D(32, 3, (3, 3), padding='VALID', activation='elu')
        self.conv3 = keras.layers.Conv2D(32, 3, (1, 1), padding='VALID', activation='elu')


class ConvModel(keras.Model):

    def __init__(self, h_size, n_actions, input_shape, ConvolutionalLayersShared=None):
        super(ConvModel, self).__init__(name="ConvQnetwork")
        if ConvolutionalLayersShared is not None:
            self.conv1 = ConvolutionalLayersShared.conv1
            self.conv2 = ConvolutionalLayersShared.conv2
            self.conv3 = ConvolutionalLayersShared.conv3
        else:
            self.conv1 = keras.layers.Conv2D(16, 3, (4, 4), padding='VALID', activation='elu')
            self.conv2 = keras.layers.Conv2D(32, 3, (4, 4), padding='VALID', activation='elu')
            self.conv3 = keras.layers.Conv2D(32, 3, (2, 2), padding='VALID', activation='elu')

        self.conv4 = keras.layers.Conv2D(32, 2, (2, 2), padding='VALID', activation='elu')
        self.flat1 = keras.layers.Flatten()
        self.dense = keras.layers.Dense(h_size, activation='elu')
        #self.Qout = keras.layers.Dense(n_actions)
        # Have the network estimate the Advantage function as an intermediate layer
        self.A = keras.layers.Dense(n_actions + 1, activation='linear')
        self.Qout = keras.layers.Lambda(lambda i: keras.backend.expand_dims(i[:, 0], -1) + i[:, 1:] - keras.backend.mean(i[:, 1:], keepdims=True),output_shape=(n_actions,))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flat1(x)
        x = self.dense(x)
        x = self.A(x)
        x = self.Qout(x)
        return x

class QnetworkEager():

    def __init__(self, input_shape, h_size, n_actions, scope_var, device, model, ConvolutionalLayersShared=None):

        tf.set_random_seed(1)
        with tf.device(device):
            if ConvolutionalLayersShared is None:
                self.model = model(h_size, n_actions, input_shape)
            else:
                self.model = model(h_size, n_actions, input_shape, ConvolutionalLayersShared)

            dummy_x = tf.zeros([1] + input_shape[1::])
            self.model._set_inputs(dummy_x)
            self.optimizer = tf.train.RMSPropOptimizer(0.0001)
            self.global_step = tf.Variable(0)

    def prediction(self, sess, s):

        s = np.array(s, dtype=np.float32)

        a = self.model(s)

        return np.argmax(a, 1)

    def Qprediction(self, sess, s):

        s = np.array(s, dtype=np.float32)

        return self.model(s).numpy()

    def grad(self, model, inputs, targets, weights):

        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss_value = Losses.huber_loss_importance_weight(outputs, targets, weights)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def train(self, sess, s, targetQ, imp_w, batch=32):

        s = np.array(s, dtype=np.float32)

        loss_value, grads = self.grad(self.model, s, targetQ, imp_w)

        # print("Step: {}, Initial Loss: {}".format(self.global_step.numpy(),loss_value.numpy()))

        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables), self.global_step)

        # loss_value, _ = self.grad(self.model, s, targetQ, imp_w)

        # print("Step: {},         Loss: {}".format(self.global_step.numpy(), loss_value.numpy()))

        return [None, None]