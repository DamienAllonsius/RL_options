import scipy.misc
from agent.SumTree import SumTree
import random
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np


class Preprocessing:

    def __init__(self, image_width, image_height, image_depth):
        self.image_width = image_width
        self.image_height = image_height
        self.image_depth = image_depth

    def preprocess_image(self, img):
        img = scipy.misc.imresize(img, ( self.image_width, self.image_height), interp='nearest')

        img = img / 255
        return img


#  Slightly modified from https://github.com/jaromiru
class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.5

    absolute_error_upper = 1.

    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _get_priority(self, error):
        return (error + self.e) ** self.a

    def add(self, sample):  # removed error argument, to new experience we give always the max
        #p = self._get_priority(error)
        #self.tree.add(p, sample)

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, sample)  # set the max p for new p

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        batch = []

        b_ISWeights = np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total() / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total()
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total()

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            batch.append((index, data))

        return batch, b_ISWeights

    def update(self, idx, error):
        clipped_errors = min(error, self.absolute_error_upper)  # clipping the error is this right?
        p = self._get_priority(clipped_errors)
        self.tree.update(idx, p)

'''
    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch
'''


class AnaliseResults:

    @staticmethod
    def reward_over_episodes(x, y):

        plt.plot(x, y)
        plt.show()

    @staticmethod
    def save_data(path_folder, file_name, data):
        directory = os.path.dirname(path_folder)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(path_folder+file_name, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_data(path_folder, file_name):

        with open(path_folder+file_name, 'rb') as f:
            mynewlist = pickle.load(f)

        return mynewlist


class UpdateWeightsModels:

    def __init__(self, weights, model):
        self.operation = []
        self.model = model
        self.weights = weights

    def _set_update(self):
        for ln1, ln2 in (zip(self.weights, self.model)):
            self.operation.append(ln2.assign(ln1))

    def set_operations(self):
        self._set_update()
        print("OPERATION TO UPDATE WEIGHTS SET UP")

    def update(self, sess):
        for op in self.operation:
            sess.run(op)

    def get_weights(self):
        return self.weights

    def get_model(self):
        return self.model

    def set_weights(self, weights):
        self.weights = weights

    def set_model(self, model):
        self.model = model


class AnalyzeMemory:

    memory_distribution = []
    reward_distribution = []

    def add_batch(self, batch):
        for i in range(len(batch)):
            episode_n = batch[i][1][-1]
            r = batch[i][1][2]
            self.memory_distribution.append(episode_n)
            self.reward_distribution.append(r)

    def plot_memory_distribution(self):
        print(len(self.memory_distribution))
        plt.hist(self.memory_distribution)
        plt.show()
        plt.hist(self.reward_distribution)
        plt.show()
