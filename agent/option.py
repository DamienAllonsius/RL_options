"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from gridenvs.utils import Direction, Point
from agent.q import Q
import time
import numpy as np
from variables import *
import random
from agent.Models import *
from agent.Utils import *
"""
TODO : all the Option classes have now to inherit from OptionAbstract class.
"""

class OptionAbstract(object):
    """
    Abstract option class that barely only needs an update function and an act function
    """
    def __init__(self, initial_state, terminal_state, play = False):
        self.initial_state = initial_state
        self.terminal_state = terminal_state
        self.play = play

    def __eq__(self, other_option):
        if type(other_option).__name__ == self.__class__.__name__:
            return (self.initial_state == other_option.initial_state) and (self.terminal_state == other_option.terminal_state)
        
        else:
            return False

    def __hash__(self):
        return hash((self.initial_state, self.terminal_state))
    
    def check_end_option(self, new_state):
        return new_state != self.initial_state

    def update_option(self, reward, new_position, new_state, action):
        raise Exception("Not Implemented")

    def act(self):
        raise Exception("Not Implemented")


class OptionDQN(OptionAbstract):

    random.seed(1)

    exp = 0
    epsilon = 1
    option_index=0

    def __init__(self, position, initial_state, terminal_state, grid_size_option, action_space, state_dimension, buffer_size,epsilon_step,
                update_target_freq, gamma, batch_size, MIN_EPSILON,tf_sess, play = False):
        super().__init__(initial_state, terminal_state, play)

        self.grid_size_option = grid_size_option
        self.position = self.get_position(position)
        self.cardinal = Direction.cardinal()
        self.number_actions = len(Direction.cardinal())
        self.reward_for_agent = 0

        self.action_space = action_space
        self.state_dimension = state_dimension
        self.buffer = Memory(buffer_size)

        self.main_model_nn = DenseQnetwork([None, 1], 128, len(self.action_space), 'mainQn'+str(OptionDQN.option_index), 'cpu:0')

        self.target_model_nn = DenseQnetwork([None, 1], 128, len(self.action_space), 'targetQn'+str(OptionDQN.option_index), 'cpu:0')

        self.ID = "OptionDQN - " + str(OptionDQN.option_index)

        OptionDQN.option_index += 1

        self.epsilon_step = epsilon_step
        self.update_target_freq = update_target_freq
        self.gamma = gamma
        self.batch_size = batch_size
        self.MIN_EPSILON = MIN_EPSILON
        self.tf_sess = tf_sess

    def act(self):

        s = [self.position]

        if self.play:
            index_action = self.main_model_nn.prediction(self.tf_sess, [s])[0]
            return self.cardinal[index_action]

        else:

            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                return self.cardinal[np.random.randint(self.number_actions)]
            else:
                index_action = self.main_model_nn.prediction(self.tf_sess, [s])[0]
                return self.cardinal[index_action]

    def _get_tderror(self, batch):
        no_state = np.zeros(self.state_dimension)
        states = np.array([o[1][0] for o in batch])
        states_ = np.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        p = self.main_model_nn.Qprediction(self.tf_sess, states)
        p_ = self.main_model_nn.Qprediction(self.tf_sess, states_)
        p_target_ = self.target_model_nn.Qprediction(self.tf_sess, states_)

        x = np.zeros((len(batch),) + self.state_dimension)
        y = np.zeros((len(batch), len(self.action_space)))
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
        #print(self.buffer.ID, batch[0])
        x, y, errors = self._get_tderror(batch)
        #print(x[0], y[0])
        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.buffer.update(idx, errors[i])

        _, loss = self.main_model_nn.train(self.tf_sess, x, y, imp_w)
        return loss

    def encode_direction(self, direction):
        """
        this function encodes a direction Direction.N/S/E/W into a number, 1/2/3/4
        """
        return self.cardinal.index(direction)

    def get_position(self, point):
        """
        point is the current position on the whole grid.
        point is projected into the zone
        """
        projected_point = point % self.grid_size_option
        return projected_point.x + self.grid_size_option.x * projected_point.y

    def update_option(self, reward, new_position, new_state, action):

        encoded_new_position = self.get_position(new_position)
        if self.play:
            self.position = encoded_new_position
            return self.check_end_option(new_state)

        else:
            encoded_action = self.encode_direction(action)
            total_reward = reward + PENALTY_OPTION_ACTION
            end_option = self.check_end_option(new_state)
            self.reward_for_agent += total_reward

            if end_option:
                if new_state == self.terminal_state:
                    total_reward += REWARD_END_OPTION

                else:
                    total_reward += PENALTY_END_OPTION

        self.observe(([self.position], encoded_action, total_reward, [encoded_new_position]))

        self.replay()

        self.position = encoded_new_position

        return end_option
     
class Option(object):
    """
    This class is doing Q learning, where Q is a matrix (we know the number of states and actions)
    """
    option_index = 0

    def __init__(self, position, initial_state, terminal_state, grid_size_option, play):
        """
        here grid_size_option is the size of the zone
        """
        self.play = play
        self.grid_size_option = grid_size_option
        self.number_state = grid_size_option.x * grid_size_option.y
        self.number_actions = len(Direction.cardinal())
        self.q = np.zeros((self.number_state, self.number_actions))
        self.cardinal = Direction.cardinal()
        self.position = self.get_position(position)
        self.initial_state = initial_state
        self.terminal_state = terminal_state
        self.reward_for_agent = 0

        self.ID = "Option - " + str(Option.option_index)

        Option.option_index += 1

    def __repr__(self):
        return "".join(["Option(", str(self.initial_state), ",", str(self.terminal_state), ")"])
    
    def __str__(self):
        return "option from " + str(self.initial_state) + " to " + str(self.terminal_state)

    def __eq__(self, other_option):
        if type(other_option).__name__ == self.__class__.__name__:
            return (self.initial_state == other_option.initial_state) and (self.terminal_state == other_option.terminal_state)
        
        else:
            return False

    def __hash__(self):
        return hash((self.initial_state, self.terminal_state))

    def check_end_option(self, new_state):
        return new_state != self.initial_state
    
    def get_position(self, point):
        """
        point is the current position on the whole grid.
        point is projected into the zone
        """
        projected_point = point % self.grid_size_option
        return projected_point.x + self.grid_size_option.x * projected_point.y
        
    def encode_direction(self, direction):
        """
        this function encodes a direction Direction.N/S/E/W into a number, 1/2/3/4
        """
        return self.cardinal.index(direction)
        
    def update_option(self, reward, new_position, new_state, action):
        encoded_new_position = self.get_position(new_position)
        if self.play:
            self.position = encoded_new_position
            return self.check_end_option(new_state)

        else:
            encoded_action = self.encode_direction(action)
            max_value_action = np.max(self.q[encoded_new_position])
            total_reward = reward + PENALTY_OPTION_ACTION
            end_option = self.check_end_option(new_state)
            self.reward_for_agent += total_reward
            if end_option:
                if new_state == self.terminal_state:
                    total_reward += REWARD_END_OPTION
                    
                else:
                    total_reward += PENALTY_END_OPTION

            self.q[self.position, encoded_action] *= (1 - LEARNING_RATE)
            self.q[self.position, encoded_action] += LEARNING_RATE * (total_reward + max_value_action)
            self.position = encoded_new_position
            return end_option
      
    def act(self):
        if self.play:
            best_action = np.argmax(self.q[self.position])

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                best_action = np.random.randint(self.number_actions)
            
            else:
                best_action = np.argmax(self.q[self.position])
            
        return self.cardinal[best_action]

class OptionExplore(object):
    """
    This is a special option to explore. No q_function is needed here.
    """
    option_index = 0

    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.reward_for_agent = 0

        self.ID = "OptionExplore - " + str(OptionExplore.option_index)

        OptionExplore.option_index += 1

    def __str__(self):
        return "explore option from " + str(self.initial_state)

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __hash__(self):
        return hash("explore")

    def act(self):
        # For the moment we do a stupid thing: go random, until it finds a new zone
        direction_number = np.random.randint(4)
        cardinal = Direction.cardinal()
        return cardinal[direction_number]
    
    def check_end_option(self, new_state):
        """
        option ends iff it has found a new zone
        """
        return new_state != self.initial_state

    def update_option(self, reward, new_position, new_state, action):
        print("option state " + str(self.initial_state))
        print("option new_state " + str(new_state))
        self.reward_for_agent += PENALTY_OPTION_ACTION
        return self.check_end_option(new_state)


class OptionExploreQ(Option):

    option_index = 0

    def __init__(self, position, initial_state, grid_size_option, last_action):
        self.grid_size_option = grid_size_option
        self.number_state = grid_size_option.x * grid_size_option.y
        self.number_actions = len(Direction.cardinal())
        self.cardinal = self.get_cardinal(last_action)
        self.position = self.get_position(position)
        self.initial_state = initial_state
        self.reward_for_agent = 0
        self.q = {}
        self.exploration_terminated = {}

        self.ID = self.ID = "OptionExploreQ - " + str(OptionExploreQ.option_index)

        OptionExploreQ.option_index += 1

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __str__(self):
        return "explore option with Q function from " + str(self.initial_state)
    
    def get_cardinal(self, permutation = 0):
        """
        We want to optimize the exploration by starting exploring in the same direction which we entered in the zone
        when permutation is
        _ 0 (= North) -> cardinal = [North, East, West, South]
        _ 1 (= East) -> cardinal = [East, South, North, West]
        _ 2 (= South) -> cardinal = [South, East, West, North]
        _ 3 (= West) -> cardinal = [West, North, South, East]

        this can be obtained by permuting [North, East, South, West] with a cycle
        and then transposing the last two elements of the list.
        """
        #return Direction.cardinal()
        cardinal = Direction.cardinal()
        permutated_cardinal = []
        for k in range(self.number_actions):
            permutated_cardinal.append(cardinal[(k + permutation) % self.number_actions])

        permutated_cardinal[3], permutated_cardinal[2] = permutated_cardinal[2], permutated_cardinal[3]
        return permutated_cardinal
    
    def update_option(self, reward, new_position, new_state, action):
        encoded_new_position = self.get_position(new_position)
        encoded_action = self.encode_direction(action)
        max_value_action = np.max(self.q[self.initial_state][self.position])
        total_reward = PENALTY_OPTION_ACTION 
        end_option = self.check_end_option(new_state)
        self.reward_for_agent += PENALTY_OPTION_ACTION
        self.q[self.initial_state][self.position, encoded_action] += total_reward
        self.set_exploration_terminated()
        self.position = encoded_new_position
        return end_option

    def set_exploration_terminated(self):
        """
        the exploration is terminated if for ALL states, the actions are : 
        - either [0, 0, 0, 0] (this would correspond to a wall for example)
        - either [-1, -3, -4, -11] (all the actions have been tried)
        """
        if not(self.exploration_terminated[self.initial_state]):
            # change only if it is false. Otherwise leave it at True
            for actions in self.q[self.initial_state]:
               terminated = (actions == [0, 0, 0, 0]).all() or (0 not in actions)
               if not(terminated):
                   self.exploration_terminated[self.initial_state] = False
                   return
            self.exploration_terminated[self.initial_state] = True
            print("exploration done -> state " + str(self.initial_state))
            
    def act(self):
        current_q_function = self.q[self.initial_state]
        max_value_action = np.argmax(current_q_function[self.position])
        return self.cardinal[max_value_action]
