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
            lost_life = (self.lives > remaining_lives)
            total_reward = self.compute_total_reward(reward, end_option, new_state_blurred, lost_life)
            
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
#            self.reward_for_agent += PENALTY_LOST_LIFE
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

#        if self.lives > remaining_lives:
#            self.reward_for_agent += PENALTY_LOST_LIFE
            
        self.reward_for_agent += reward # the option shows a sample of the possible reward of the state to the agent
        self.lives = remaining_lives
        return self.check_end_option(new_state_blurred)
