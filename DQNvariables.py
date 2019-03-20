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

MEMORY_CAPACITY = 10000
EXPLORATION_STOP = 1000
EPSILON_DECAY_RATE = 1 / EXPLORATION_STOP
UPDATE_TARGET_FREQ = 10
DEVICE = 'cpu:0'
ACTION_SPACE = [1, 2, 3, 4]#[0, 1]
GAMMA = 0.99
BATCH_SIZE = 32
MIN_EPSILON = 0.01
TF_SESSION = None

state_dimension = (1,)
input_shape_nn = [None, 1]

#buffer = Memory(MEMORY_CAPACITY)
#main_model_nn = None #Linear(input_shape_nn, 128, len(ACTION_SPACE), 'mainQn', DEVICE)
#target_model_nn = None #Linear(input_shape_nn, 128, len(ACTION_SPACE), 'targetQn', DEVICE)
#trainables_mainQN = tf.trainable_variables('mainQn')
#trainables_targetQN = tf.trainable_variables('targetQn')
#upd_target = UpdateWeightsModels(trainables_mainQN, trainables_targetQN)
#upd_target.set_operations()

tf_sess = None #tf.Session()
#tf_sess.run(tf.global_variables_initializer())
