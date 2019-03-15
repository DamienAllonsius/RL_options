"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from gridenvs.utils import Point
from agent.q import QArray
import time
import numpy as np
from variables import *

class Option(object):
    """
    This class is doing Q learning, where Q is a matrix (we know the number of states and actions)
    """
    def __init__(self, number_actions, initial_state, current_state, terminal_state, play):
        """
        here grid_size_option is the size of the zone 
        state are always of high resolution except if stated otherwise in the variable name
        """
        self.play = play
        self.number_actions = number_actions
        self.q = QArray(current_state, number_actions)
        self.initial_state = initial_state # blurred image
        self.current_state = current_state # high resolution image
        self.terminal_state = terminal_state # blurred image
        self.reward_for_agent = 0 # the positive rewards received by the environment
        self.lives = None

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
        """
        states are tuples
        """
        return self.initial_state + self.terminal_state

    def check_end_option(self, new_state_blurred):
        return new_state_blurred != self.initial_state

    def set_current_state(self, current_state):
        self.current_state = current_state
        self.q.add_state(current_state)
    
    def update_option(self, reward, new_state, action, remaining_lives):            
        if self.lives == None:
            self.lives = remaining_lives
            
        end_option = self.check_end_option(new_state["blurred_state"])

        if self.play:
            return end_option

        else:
            self.reward_for_agent += reward
            lost_life = (self.lives > remaining_lives)
            total_reward = self.compute_total_reward(reward, end_option, new_state["blurred_state"], lost_life)
            
            self.q.update_q_action_state(self.current_state, new_state["state"], action)
            self.q.update_q_value(self.current_state, action, total_reward, new_state["state"], end_option)
            self.lives = remaining_lives
            self.current_state = new_state["state"]
            return end_option
      
    def compute_total_reward(self, reward, end_option, new_state_blurred, lost_life):
        total_reward = reward + PENALTY_OPTION_ACTION
        if end_option:
            if new_state_blurred == self.terminal_state:
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
                best_action = self.q.get_random_action(self.current_state)
            
            else:
                _, best_action = self.q.find_best_action(self.current_state)

            
        return best_action

class OptionExplore(object):
    """
    This is a special option to explore. No q_function is needed here.
    """
    def __init__(self, initial_state, number_actions):
        self.initial_state = initial_state
        self.reward_for_agent = 0
        self.number_actions = number_actions
        self.lives = None

    def __str__(self):
        return "explore option from " + str(self.initial_state)

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __hash__(self):
        return hash("explore")

    def act(self):
        # here we do a stupid thing: go random, until it finds a new zone
        return (np.random.randint(self.number_actions))
    
    def check_end_option(self, new_state_blurred):
        """
        option ends iff it has found a new zone
        """
        return new_state_blurred != self.initial_state

    def update_option(self, reward, new_state, action, remaining_lives):
        if self.lives == None:
            self.lives = remaining_lives

#        if self.lives > remaining_lives:
#            self.reward_for_agent += PENALTY_LOST_LIFE
            
        self.reward_for_agent += reward # the option shows a sample of the possible reward of the state to the agent
        self.lives = remaining_lives
        return self.check_end_option(new_state["blurred_state"])

# class OptionExploreQ(Option):

#     def __init__(self, position, initial_state, grid_size_option):
#         self.grid_size_option = grid_size_option
#         self.number_state = grid_size_option.x * grid_size_option.y
#         self.number_actions = len(Direction.cardinal())
#         self.position = self.get_position(position)
#         self.initial_state = initial_state
#         self.reward_for_agent = 0
#         self.q = {}
#         self.exploration_terminated = {}

#     def __eq__(self, other):
#         return type(other).__name__ == self.__class__.__name__

#     def __str__(self):
#         return "explore option with Q function from " + str(self.initial_state)
    
#     def update_option(self, reward, new_position, new_state, action):
#         encoded_new_position = self.get_position(new_position)
#         max_value_action = np.max(self.q[self.initial_state][self.position])
#         total_reward = PENALTY_OPTION_ACTION 
#         end_option = self.check_end_option(new_state)
#         self.reward_for_agent += PENALTY_OPTION_ACTION
#         self.q[self.initial_state][self.position, action] += total_reward
#         self.set_exploration_terminated()
#         self.position = encoded_new_position
#         return end_option

#     def set_exploration_terminated(self):
#         """
#         the exploration is terminated if for ALL states, the actions are : 
#         - either [0, 0, 0, 0] (this would correspond to a wall for example)
#         - either [-1, -3, -4, -11] (all the actions have been tried)
#         """
#         if not(self.exploration_terminated[self.initial_state]):
#             # change only if it is false. Otherwise leave it at True
#             for actions in self.q[self.initial_state]:
#                terminated = (actions == [0, 0, 0, 0]).all() or (0 not in actions)
#                if not(terminated):
#                    self.exploration_terminated[self.initial_state] = False
#                    return
               
#             self.exploration_terminated[self.initial_state] = True
#             print("exploration done -> state " + str(self.initial_state))
            
#     def act(self):
#         current_q_function = self.q[self.initial_state]
#         max_value_action = np.argmax(current_q_function[self.position])
#         return max_value_action

