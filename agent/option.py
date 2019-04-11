"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from gridenvs.utils import Point
from agent.q import QDict, QArray
import time
import numpy as np
from variables import *
import random
from agent.Models import *
from agent.Utils import *


class OptionAbstract(object):
    """
    Abstract option class that barely only needs an update function and an act function
    """

    def __init__(self, initial_state_blurred, terminal_state_blurred, play=False):
        self.initial_state_blurred = initial_state_blurred
        self.terminal_state_blurred = terminal_state_blurred
        self.play = play

    def __repr__(self):
        return "".join(["Option(", str(self.initial_state_blurred), ",", str(self.terminal_state_blurred), ")"])

    def __str__(self):
        return "option from " + str(self.initial_state_blurred) + " to " + str(self.terminal_state_blurred)

    def __eq__(self, other_option):
        if type(other_option).__name__ == self.__class__.__name__:
            return (self.initial_state_blurred == other_option.initial_state_blurred) and (
                        self.terminal_state_blurred == other_option.terminal_state_blurred)

        else:
            return False

    def __hash__(self):
        """
        states are tuples
        """
        return self.initial_state_blurred + self.terminal_state_blurred

    def check_end_option(self, new_state_blurred):
        return new_state_blurred != self.initial_state_blurred

    def update_option(self, reward, new_position, new_state, action):
        raise Exception("Not Implemented")

    def act(self):
        raise Exception("Not Implemented")


class OptionDQN(OptionAbstract):
    random.seed(1)

    exp = 0
    epsilon = 1
    option_index = 0

    def __init__(self, number_actions, initial_state_blurred, current_state, terminal_state_blurred, state_dimension, input_shape,
                 buffer_size, epsilon_step, update_target_freq, gamma, batch_size, MIN_EPSILON, tf_sess, conv_shared_main_model,
                 conv_shared_target_model, play=False):
        super().__init__(initial_state_blurred, terminal_state_blurred, play)

        self.number_actions = number_actions
        self.current_state = current_state # high resolution image
        self.reward_for_agent = 0
        self.lives = None

        self.state_dimension = state_dimension
        self.buffer = Memory(buffer_size)

        self.main_model_nn = QnetworkEager(input_shape, 256, self.number_actions,
                                           'mainQn' + str(OptionDQN.option_index), 'cpu:0', ConvModel, conv_shared_main_model)

        self.target_model_nn = QnetworkEager(input_shape, 256, self.number_actions,
                                             'targetQn' + str(OptionDQN.option_index), 'cpu:0', ConvModel, conv_shared_target_model)

        self.ID = "OptionDQN - " + str(OptionDQN.option_index)

        OptionDQN.option_index += 1

        self.epsilon_step = epsilon_step
        self.update_target_freq = update_target_freq
        self.gamma = gamma
        self.batch_size = batch_size
        self.MIN_EPSILON = MIN_EPSILON
        self.tf_sess = tf_sess

    def act(self):

        s = self.current_state

        if self.play:
            index_action = self.main_model_nn.prediction(self.tf_sess, [s])[0]
            return index_action

        else:

            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                return np.random.randint(self.number_actions)
            else:
                index_action = self.main_model_nn.prediction(self.tf_sess, [s])[0]
                return index_action

    def _get_tderror(self, batch):
        no_state = np.zeros(self.state_dimension)
        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.main_model_nn.Qprediction(self.tf_sess, states)
        p_ = self.main_model_nn.Qprediction(self.tf_sess, states_)
        p_target_ = self.target_model_nn.Qprediction(self.tf_sess, states_)

        x = np.zeros((len(batch),) + self.state_dimension)
        y = np.zeros((len(batch), self.number_actions))
        errors = np.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]
            a = o[1]
            r = o[2]
            s_ = o[3]

            t = p[i]
            a_index = a
            old_val = t[a_index]
            if s_ is None:
                t[a_index] = r
            else:
                t[a_index] = r + self.gamma * p_target_[i][np.argmax(p_[i])]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(old_val - t[a_index])

        return x, y, errors

    def observe(self, sample):  # in (s, a, r, s_) format
        self.buffer.add(sample)
        if self.exp % self.update_target_freq == 0:
            self.target_model_nn.model.set_weights(self.main_model_nn.model.get_weights())

        # slowly decrease Epsilon based on our experience
        self.exp += 1
        if self.epsilon > self.MIN_EPSILON:
            self.epsilon -= self.epsilon_step

    def replay(self):
        batch, imp_w = self.buffer.sample(self.batch_size)
        # print(self.buffer.ID, batch[0])
        x, y, errors = self._get_tderror(batch)
        # print(x[0], y[0])
        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.buffer.update(idx, errors[i])

        _, loss = self.main_model_nn.train(self.tf_sess, x, y, imp_w)
        return loss

    def compute_total_reward(self, reward, end_option, new_state_blurred, lost_life):
        total_reward = reward + PENALTY_OPTION_ACTION
        if end_option:
            if new_state_blurred == self.terminal_state_blurred:
                total_reward += REWARD_END_OPTION
                print("option terminated correctly")

            else:
                total_reward += PENALTY_END_OPTION
                print("missed")

        if lost_life:
            self.reward_for_agent += PENALTY_LOST_LIFE
            total_reward += PENALTY_LOST_LIFE

        return total_reward

    def update_option(self, reward, new_state, new_state_blurred, action, remaining_lives):
        if self.lives is None:
            self.lives = remaining_lives

        end_option = self.check_end_option(new_state_blurred)
        if self.play:
            return end_option

        else:
            self.reward_for_agent += reward
            total_reward = self.compute_total_reward(reward, end_option, new_state_blurred,
                                                     lost_life=(self.lives > remaining_lives))

            self.observe((self.current_state, action, total_reward, new_state))

            self.replay()

            self.lives = remaining_lives

            return end_option

class Option(object):
    """
    This class is doing Q learning, where Q is a matrix (we know the number of states and actions)
    """
    def __init__(self, number_actions, initial_state_blurred, current_state, terminal_state_blurred, play):
        """
        here grid_size_option is the size of the zone 
        state are always of high resolution except if stated otherwise in the variable name
        """
        self.play = play
        self.number_actions = number_actions
        self.q = QArray(current_state, number_actions)
        self.initial_state_blurred = initial_state_blurred # blurred image
        self.current_state = current_state # high resolution image
        self.terminal_state_blurred = terminal_state_blurred # blurred image
        self.reward_for_agent = 0
        self.lives = None

    def __repr__(self):
        return "".join(["Option(", str(self.initial_state_blurred), ",", str(self.terminal_state_blurred), ")"])
    
    def __str__(self):
        return "option from " + str(self.initial_state_blurred) + " to " + str(self.terminal_state_blurred)

    def __eq__(self, other_option):
        if type(other_option).__name__ == self.__class__.__name__:
            return (self.initial_state_blurred == other_option.initial_state_blurred) and (self.terminal_state_blurred == other_option.terminal_state_blurred)
        
        else:
            return False

    def __hash__(self):
        """
        states are tuples
        """
        return self.initial_state_blurred + self.terminal_state_blurred

    def check_end_option(self, new_state_blurred):
        return new_state_blurred != self.initial_state_blurred
    
    def update_option(self, reward, new_state, new_state_blurred, action, remaining_lives):
        if self.lives == None:
            self.lives = remaining_lives
            
        end_option = self.check_end_option(new_state_blurred)
        if self.play:
            return end_option

        else:
            self.reward_for_agent += reward 
            total_reward = self.compute_total_reward(reward, end_option, new_state_blurred, lost_life = (self.lives > remaining_lives))
            self.q.update_q_function_action_state(self.current_state, new_state, action)
            self.q.update_q_function_value(self.current_state, action, total_reward, new_state)
            self.lives = remaining_lives
            return end_option
      
    def compute_total_reward(self, reward, end_option, new_state_blurred, lost_life):
        total_reward = reward + PENALTY_OPTION_ACTION
        if end_option:
            if new_state_blurred == self.terminal_state_blurred:
                total_reward += REWARD_END_OPTION
                print("option terminated correctly")
                
            else:
                total_reward += PENALTY_END_OPTION
                print("missed")
                
        if lost_life:
            self.reward_for_agent += PENALTY_LOST_LIFE
            total_reward += PENALTY_LOST_LIFE
            
        return total_reward

    def act(self):
        if self.play:
            _, best_action = self.q.find_best_action(self.current_state)

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                best_action = np.random.randint(self.number_actions)
            
            else:
                _, best_action = self.q.find_best_action(self.current_state)
            
        return best_action

class OptionExplore(object):
    """
    This is a special option to explore. No q_function is needed here.
    """
    def __init__(self, initial_state_blurred, number_actions):
        self.initial_state_blurred = initial_state_blurred
        self.reward_for_agent = 0
        self.number_actions = number_actions
        self.lives = None

    def __str__(self):
        return "explore option from " + str(self.initial_state_blurred)

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __hash__(self):
        return hash("explore")

    def act(self):
        # here we do a stupid thing: go random, until it finds a new zone
        return np.random.randint(self.number_actions)
    
    def check_end_option(self, new_state_blurred):
        """
        option ends iff it has found a new zone
        """
        return new_state_blurred != self.initial_state_blurred

    def update_option(self, reward, new_state, new_state_blurred, action, remaining_lives):
        if self.lives == None:
            self.lives = remaining_lives

        if self.lives > remaining_lives:
            self.reward_for_agent += PENALTY_LOST_LIFE
            
        self.reward_for_agent += reward # the option shows a sample of the possible reward of the state to the agent
        self.lives = remaining_lives
        return self.check_end_option(new_state_blurred)
