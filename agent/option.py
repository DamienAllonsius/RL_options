"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from gridenvs.utils import Point
from agent.q import Q
import time
import numpy as np
from variables import *

class Option(object):
    """
    This class is doing Q learning, where Q is a matrix (we know the number of states and actions)
    """
    def __init__(self, number_actions, initial_state, terminal_state, play):
        """
        here grid_size_option is the size of the zone 
        """
        self.play = play
        self.number_actions = number_actions
        self.q = Q(initial_state, number_actions)
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
    
    def update_option(self, reward, new_state, action):
        if self.play:
            return self.check_end_option(new_state)

        else:
            self.reward_for_agent += reward
            total_reward = reward + PENALTY_OPTION_ACTION
            end_option = self.check_end_option(new_state)
            if end_option:
                if new_state == self.terminal_state:
                    total_reward += REWARD_END_OPTION
                    
                else:
                    total_reward += PENALTY_END_OPTION

            self.q.q_function.update_q_function_value(self.state, action, total_reward, new_state):
            return end_option
      
    def act(self):
        if self.play:
            _, best_action = self.q.find_best_action(self.initial_state)

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_IN_OPTION:
                best_action = np.random.randint(self.number_actions)
            
            else:
                _, best_action = self.q.find_best_action(self.initial_state)
            
        return best_action

class OptionExplore(object):
    """
    This is a special option to explore. No q_function is needed here.
    """
    def __init__(self, initial_state, number_actions):
        self.initial_state = initial_state
        self.reward_for_agent = 0
        self.number_actions = number_actions

    def __str__(self):
        return "explore option from " + str(self.initial_state)

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

    def __hash__(self):
        return hash("explore")

    def act(self):
        # here we do a stupid thing: go random, until it finds a new zone
        return np.random.randint(self.number_actions)
    
    def check_end_option(self, new_state):
        """
        option ends iff it has found a new zone
        """
        return new_state != self.initial_state

    def update_option(self, reward, new_state, action):
        self.reward_for_agent += reward # the option shows a sample of the possible reward of the state to the agent
        return self.check_end_option(new_state)


class OptionExploreQ(Option):
    """
    refactoring, TODO
    """
    def __init__(self, position, initial_state, grid_size_option, last_action):
        raise Exception("not implimented yet")
        self.grid_size_option = grid_size_option
        self.number_state = grid_size_option.x * grid_size_option.y
        self.number_actions = len(Direction.cardinal())
        self.cardinal = self.get_cardinal(last_action)
        self.position = self.get_position(position)
        self.initial_state = initial_state
        self.reward_for_agent = 0
        self.q = {}
        self.exploration_terminated = {}

    def __eq__(self, other):
        return type(other).__name__ == self.__class__.__name__

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
