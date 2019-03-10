""" DQN VARIABLES """
from agent.Models import *
from agent.Utils import *
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Just to be sure that we don't have some others graph loaded
tf.reset_default_graph()



IMAGE_WIDTH = 42
IMAGE_HEIGHT = 42
IMAGE_DEPTH = 3
MEMORY_CAPACITY = 60000
PROBLEM = 'GE_MazeOptions-v1'
EXPLORATION_STOP = 400000
EPSILON_DECAY_RATE = 1 / EXPLORATION_STOP
UPDATE_TARGET_FREQ = 5000
DEVICE = 'cpu:0'
ACTION_SPACE = [1, 2, 3, 4]#[0, 1]
GAMMA = 0.99
BATCH_SIZE = 32 #256
MIN_EPSILON = 0.01
TF_SESSION = None

state_dimension = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH)
input_shape_nn = [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]

buffer = Memory(MEMORY_CAPACITY)
main_model_nn = DualQnetwork(input_shape_nn, 128, len(ACTION_SPACE), 'mainQn', DEVICE)
target_model_nn = DualQnetwork(input_shape_nn, 128, len(ACTION_SPACE), 'targetQn', DEVICE)
trainables_mainQN = tf.trainable_variables('mainQn')
trainables_targetQN = tf.trainable_variables('targetQn')
upd_target = UpdateWeightsModels(trainables_mainQN, trainables_targetQN)
upd_target.set_operations()

tf_sess = tf.Session()
tf_sess.run(tf.global_variables_initializer())
