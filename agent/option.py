"""
This class is for making options
For the moment we only implement the "exploring option"
"""
from agent.q import QArray
import numpy as np


class Option(object):
    """
    This class is doing Q learning, where Q is a matrix (we know the number of states and actions)
    """
    def __init__(self, number_actions, initial_state, current_state, terminal_state, play, experiment_data):
        """
        here grid_size_option is the size of the zone 
        state are always of high resolution except if stated otherwise in the variable name
        """
        self.experiment_data = experiment_data
        self.play = play
        self.number_actions = number_actions
        self.q = QArray(current_state, number_actions)
        self.initial_state = initial_state  # blurred image
        self.current_state = current_state  # high resolution image
        self.terminal_state = terminal_state  # blurred image
        self.reward_for_agent = 0  # the positive rewards received by the environment
        self.lost_life = False
        self.lives = None

    def __repr__(self):
        return "".join(["Option(", str(self.initial_state), ",", str(self.terminal_state), ")"])
    
    def __str__(self):
        return "option from " + str(self.initial_state) + " to " + str(self.terminal_state)

    def __eq__(self, other_option):
        if type(other_option).__name__ == self.__class__.__name__:
            return (self.initial_state == other_option.initial_state) and \
                   (self.terminal_state == other_option.terminal_state)
        
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
        if self.lives is None:
            self.lives = remaining_lives
            
        end_option = self.check_end_option(new_state["blurred_state"])
        
        if self.play:
            return end_option

        else:
            # self.reward_for_agent += reward
            self.lost_life = (self.lives > remaining_lives)
            total_reward = self.compute_total_reward(reward, end_option, new_state["blurred_state"], action)
            
            self.q.update_q_action_state(self.current_state, new_state["state"], action)
            self.q.update_q_value(self.current_state,
                                  action,
                                  total_reward,
                                  new_state["state"],
                                  end_option,
                                  self.experiment_data["LEARNING_RATE"])

            self.lives = remaining_lives
            self.current_state = new_state["state"]
            return end_option
      
    def compute_total_reward(self, reward, end_option, new_state_blurred, action):
        total_reward = reward + self.experiment_data["PENALTY_OPTION_ACTION"] * (action != 0)
        self.reward_for_agent = reward
        if end_option:
            if new_state_blurred == self.terminal_state:
                total_reward += self.experiment_data["REWARD_END_OPTION"]
                # print("option terminated correctly")
                
            else:
                total_reward += self.experiment_data["PENALTY_END_OPTION"]
                # print("missed")
                
        if self.lost_life:
            total_reward += self.experiment_data["PENALTY_LOST_LIFE_FOR_OPTIONS"]
            
        return total_reward

    def act(self):
        if self.play:
            _, best_action = self.q.find_best_action(self.current_state)

        else:
            if np.random.rand() < self.experiment_data["PROBABILITY_EXPLORE_IN_OPTION"]:
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
        self.lost_life = False

    def __str__(self):
        return "explore option from " + str(self.initial_state)

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
        return new_state_blurred != self.initial_state

    def update_option(self, reward, new_state, action, remaining_lives):
        if self.lives is None:
            self.lives = remaining_lives

        #        if self.lives > remaining_lives:
        #            self.reward_for_agent += PENALTY_LOST_LIFE
           
        self.reward_for_agent += reward  # the option shows a sample of the possible reward of the state to the agent
        self.lives = remaining_lives
        return self.check_end_option(new_state["blurred_state"])
