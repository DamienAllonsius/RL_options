"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from gridenvs.utils import Point
from agent.q import QDict, QArray
import time
import numpy as np
from variables import *

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
        return hash((self.initial_state_blurred, self.terminal_state_blurred))

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
            self.q.update_q_function_action_state(self.state, new_state, action, reward)
            self.q.update_q_function_value(self.current_state, action, total_reward, new_state)
            self.lives = remaining_lives
            return end_option
      
    def act(self):
        if self.play:
            _, best_action = self.q.find_best_action(self.current_state)

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                best_action = np.random.randint(self.number_actions)
            
            else:
                _, best_action = self.q.find_best_action(self.current_state)
            
        return best_action

    def compute_total_reward(self, reward, end_option, new_state_blurred, lost_life):
        total_reward = reward + PENALTY_OPTION_ACTION
        if end_option:
            if new_state_blurred == self.terminal_state_blurred:
                total_reward += REWARD_END_OPTION
                
            else:
                total_reward += PENALTY_END_OPTION
        if lost_life:
            print(lost_life)
            self.reward_for_agent += PENALTY_LOST_LIFE
            total_reward += PENALTY_LOST_LIFE
            
        return total_reward
                

class OptionExplore(object):
    """
    This is a special option to explore. No q_function is needed here.
    """
    def __init__(self, initial_state_blurred, number_actions):
        self.initial_state_blurred = initial_state_blurred
        self.reward_for_agent = 0
        self.number_actions = number_actions

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
        return new_state_blurred == self.initial_state_blurred

    def update_option(self, reward, new_state, new_state_blurred, action, remaining_lives):
        self.reward_for_agent += reward # the option shows a sample of the possible reward of the state to the agent
        return self.check_end_option(new_state_blurred)


class OptionExploreQ(Option):
    """
    refactoring, TODO
    """
    def __init__(self, number_actions, initial_state_blurred, current_state, last_action):
        self.number_actions = number_actions
        self.initial_state_blurred = initial_state_blurred
        self.current_state = current_state
        self.reward_for_agent = 0
        self.q = QArray(current_state)
#        self.exploration_terminated = {}

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def get_permuted_actions(self, permutation = 0):
        """
        We want to optimize the exploration by starting exploring in the same direction which we entered in the zone
        """
        permutated_cardinal = []
        permuted_actions = range(self.number_actions)
        for act in permuted_actions:
            permuted_actions.append((act + permutation) % self.number_actions)

        return permuted_actions
    
    def update_option(self, reward, new_position, new_state, action):
        encoded_new_position = self.get_position(new_position)
        encoded_action = self.encode_direction(action)
        max_value_action = np.max(self.q[self.initial_state][self.position])
        total_reward = PENALTY_OPTION_ACTION + reward
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
