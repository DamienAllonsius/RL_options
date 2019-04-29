""" DQN VARIABLES """
from agent.Models import *
from agent.Utils import *
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.enable_eager_execution()

# Just to be sure that we don't have some others graph loaded
tf.reset_default_graph()

MEMORY_CAPACITY = 1000
EXPLORATION_STOP = 1000
EPSILON_DECAY_RATE = 1 / EXPLORATION_STOP
UPDATE_TARGET_FREQ = 1000
DEVICE = 'cpu:0'
ACTION_SPACE = [0, 1, 2, 3]
GAMMA = 0.99
BATCH_SIZE = 32
MIN_EPSILON = 0.01
TF_SESSION = None

state_dimension =(210, 160, 3) #(84, 84, 3)
input_shape_nn =[None, 210, 160, 3] #[None, 84, 84, 3]

tf_sess = None

conv_shared_main_model = ConvolutionalLayersShared() #None
conv_shared_target_model = ConvolutionalLayersShared() #None