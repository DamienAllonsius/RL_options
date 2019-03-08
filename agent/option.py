"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from gridenvs.utils import Direction, Point
from agent.q import QArray
import time
import numpy as np
from variables import *

class Option(object):
    """
    This class is doing Q learning, where Q is a matrix (we know the number of states and actions)
    """
    def __init__(self, position, initial_state, terminal_state, grid_size_option, play):
        """
        here grid_size_option is the size of the zone 
        """
        self.play = play
        self.grid_size_option = grid_size_option
        self.number_states = grid_size_option.x * grid_size_option.y
        self.number_actions = len(Direction.cardinal())
        self.q = QArray(self.number_states, self.number_actions)
        self.position = self.get_position(position)
        self.initial_state = initial_state
        self.terminal_state = terminal_state
        self.reward_for_agent = 0

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
        
    def update_option(self, reward, new_position, new_state, action):
        encoded_new_position = self.get_position(new_position)
        if self.play:
            self.position = encoded_new_position
            return self.check_end_option(new_state)

        else:
            end_option = self.check_end_option(new_state)
            self.reward_for_agent += reward + PENALTY_OPTION_ACTION
            total_reward = self.compute_total_reward(reward, new_state, end_option)
            self.q.update_q_value(self.position, action, total_reward, encoded_new_position, end_option)
            self.position = encoded_new_position
            return end_option

    def compute_total_reward(self, reward, new_state, end_option):
        total_reward = reward + PENALTY_OPTION_ACTION
        if end_option:
            if new_state == self.terminal_state:
                total_reward += REWARD_END_OPTION
                
            else:
                total_reward += PENALTY_END_OPTION

        return total_reward
        
    def act(self):
        if self.play:
            _, best_action = self.q.find_best_action(self.position)

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                best_action = self.q.get_random_action(self.position)
            
            else:
                _, best_action = self.q.find_best_action(self.position)
            
        return best_action

class OptionExplore(object):
    """
    This is a special option to explore. No q_function is needed here.
    """
    def __init__(self, initial_state):
        self.initial_state = initial_state
        self.reward_for_agent = 0

    def __str__(self):
        return "explore option from " + str(self.initial_state)

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __hash__(self):
        return hash("explore")

    def act(self):
        # For the moment we do a stupid thing: go random, until it finds a new zone
        return np.random.randint(4)
    
    def check_end_option(self, new_state):
        """
        option ends iff it has found a new zone
        """
        return new_state != self.initial_state

    def update_option(self, reward, new_position, new_state, action):
        self.reward_for_agent += PENALTY_OPTION_ACTION
        return self.check_end_option(new_state)


class OptionExploreQ(Option):

    def __init__(self, position, initial_state, grid_size_option):
        self.grid_size_option = grid_size_option
        self.number_state = grid_size_option.x * grid_size_option.y
        self.number_actions = len(Direction.cardinal())
        self.position = self.get_position(position)
        self.initial_state = initial_state
        self.reward_for_agent = 0
        self.q = {}
        self.exploration_terminated = {}

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __str__(self):
        return "explore option with Q function from " + str(self.initial_state)
    
    def update_option(self, reward, new_position, new_state, action):
        encoded_new_position = self.get_position(new_position)
        max_value_action = np.max(self.q[self.initial_state][self.position])
        total_reward = PENALTY_OPTION_ACTION 
        end_option = self.check_end_option(new_state)
        self.reward_for_agent += PENALTY_OPTION_ACTION
        self.q[self.initial_state][self.position, action] += total_reward
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
        return max_value_action
