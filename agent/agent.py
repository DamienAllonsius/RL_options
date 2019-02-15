""" This is an abstract class for agents"""

from gridenvs.keyboard_controller import Controls, Key
from gridenvs.utils import Point
import numpy as np
import time
from agent.option import Option, OptionExplore, OptionExploreQ
from agent.q import QDict, QArray
from variables import *
from data.save import SaveData
"""
TODO for pix. 

Some ideas: 

the agent state can be easily replaced by a set of pixels (N.B: make a function to compute the zones in main.py)
for the option, we should first go to a previous commit to set the q function as a dict as before.
No more tabular q function in the case of a pixel based policy.
The projection of the point should be the projection of the image to the zone of interest.
"""

class AgentOption(): 

    def __init__(self, state_blurred, number_actions, type_exploration, play):
        self.number_actions = number_actions
#        self.last_action = 0 #last action : north, east, south, west ?
        self.play = play
        self.state_blurred = state_blurred
        self.q = QDict(self.state_blurred)
        self.reward = 0
        self.type_exploration = type_exploration
        if not(play):
            if type_exploration == "OptionExploreQ":
                raise Exception("OptionExploreQ not implemented yet")
                self.explore_option = OptionExploreQ(self.state, last_action = self.last_action) # special explore options
            elif type_exploration == "OptionExplore":
                self.explore_option = OptionExplore(self.state_blurred, self.number_actions) # special explore options
            else:
                raise Exception("type_exploration unknown")

    def make_save_data(self, seed):
        #TODO : monitor ?
        self.save_data = SaveData("data/options/data_reward_" + self.__class__.__name__, seed)
        
    def reset_explore_option(self):
        self.explore_option.reward_for_agent = 0
        self.explore_option.initial_state_blurred = self.state_blurred
        if type(self.explore_option).__name__ == "OptionExploreQ":
            raise Exception("OptionExploreQ not implemented yet")
        
        """
        TODO : refactor this
        """
        # self.explore_option.position = self.explore_option.get_position(self.position)
        # self.explore_option.cardinal = self.explore_option.get_cardinal(self.last_action)
        # # if this state is explored for the first time : make a new q function full of zeros
        # # and set the exploration_terminated boolean for this state to False
        # try:
        #     self.explore_option.q[self.state]
        # except:
        #     self.explore_option.q.update({self.state : np.zeros((self.explore_option.number_state, self.explore_option.number_actions))})
        #     self.explore_option.exploration_terminated.update({self.state : False})
        
    def reset(self, initial_agent_state_blurred):
        """
        Same as __init__ but the q function is preserved 
        """
        self.reward = 0
        self.state_blurred = initial_agent_state_blurred
        self.reset_explore_option()

    def choose_option(self, t):
        """
        if no option : explore
        else flip a coin, then take the best or explore
        """
        if self.play: # in this case we do not learn anymore
            _, best_option = self.q.find_best_action(self.state_blurred)
            best_option.play = True
            return best_option

        else:
            # No option available : explore, and do not count the number of explorations
            if not(self.q.is_actions(self.state_blurred)): 
                self.reset_explore_option()
                return self.explore_option
            
            # options are available : if the exploration is not done then continue exploring
            elif ((self.type_exploration == "OptionExploreQ" and
                   not(self.explore_option.exploration_terminated[self.state_blurred])) or
                  (self.type_exploration == "OptionExplore" and
                   not(self.explore_option.check_end_option(self.state_blurred)))):
                self.reset_explore_option()
                return self.explore_option
        
            # in this case find the best option
            else:
                best_reward, best_option = self.q.find_best_action(self.state_blurred)
                if best_reward == 0:
                    best_option = self.q.get_random_action(self.state_blurred)
                    best_option.reward_for_agent = 0
                    return best_option
                
                else:
                    best_option.reward_for_agent = 0
                    return best_option

    def update_agent(self, new_state, new_state_blurred, option, action):
        """
        In this order
        _update last action done
        _update reward
        _add an option if a new state has just been discovered
        _update the q function value
        _update the state
        """
        if self.play:
            self.state_blurred = new_state_blurred
            
        else:
#            self.last_action = action
            total_reward = PENALTY_AGENT_ACTION + option.reward_for_agent
            self.reward += option.reward_for_agent
            self.update_q_function_options(new_state, new_state_blurred, option, total_reward)
            self.state_blurred = new_state_blurred
            
    def update_q_function_options(self, new_state, new_state_blurred, option, reward):
        #if self.no_return_update(new_state): #update or not given if the reverse option already exists
        action = Option(self.number_actions, self.state_blurred, new_state, new_state_blurred, self.play)
        # if the state and the action already exist, this line will do nothing
        self.q.update_q_function_action_state(self.state_blurred, new_state_blurred, action, reward)
        if option != self.explore_option:
            self.q.update_q_function_value(self.state_blurred, option, reward, new_state_blurred)

    # def no_return_update(self, new_state):
    #     """[
    #     (no return option)
    #         does not add anything if 
    #         for action in q[option.terminal_state]:
    #         action.terminal_state = option.initial_state
    #     """
    #     if self.q.is_state(new_state):
    #         new_state_idx = self.q.state_list.index(new_state)
    #         for action in self.q.q_function[new_state_idx]:
    #             if np.array_equal(action.terminal_state, self.state):
    #                 return False

    #     return True
    
    def record_reward(self, t):
        """
        save the reward in a file following this pattern:
        iteration_1 reward_1
        iteration_2 reward_2
        iteration_3 reward_3
        """
        self.save_data.record_data(t, self.reward)

class KeyboardAgent(object):
    def __init__(self, env, controls={**Controls.Arrows, **Controls.KeyPad}):
        self.env = env
        self.env.render_scaled()
        self.human_wants_shut_down = False
        self.controls = controls
        self.human_agent_action = None
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release

    def key_press(self, key, mod):
        if key==Key.esc:
            self.human_wants_shut_down = True
            
        elif key in self.controls.keys():
            self.human_agent_action = self.controls[key]
            
        else:
            raise Exception("Key %d not in controls map %s"%(key, self.controls))
    
    def key_release(self, key, mod):
        pass
    
    def act(self):
        action = self.human_agent_action
        self.human_agent_action = None
        return action


class QAgent(object):
    def __init__(self, position, number_actions, grid_size, play):
        self.play = play
        self.grid_size = grid_size
        self.number_state = 2 * grid_size.x * grid_size.y + 1
        self.number_actions = number_actions
        self.q = np.zeros((self.number_state, self.number_actions))
        self.state_id = 0
        self.position = self.encode_position(position)
        self.reward = 0

    def make_save_data(self, seed):
        self.save_data = SaveData("data/QAgent/data_reward_" + self.__class__.__name__, seed)
       
    def encode_position(self, point):
        """
        this function encodes the state from a point to a number
        """
        if self.state_id < 2:
            return point.x + self.grid_size.x * point.y + self.grid_size.x * self.grid_size.y * self.state_id
        else:
            return self.number_state - 1
        
    def reset(self, initial_position):
        self.reward = 0
        self.position = initial_position

    def update(self, reward, new_position, action, new_state_id):
        encoded_action = self.encode_direction(action)
        encoded_position = self.encode_position(new_position)
        max_value_action = np.max(self.q[encoded_position])
        total_reward = reward + PENALTY_ACTION
        self.reward += total_reward
        self.q[self.position, encoded_action] *= (1 - LEARNING_RATE)
        self.q[self.position, encoded_action] += LEARNING_RATE * (total_reward + max_value_action)
        self.position = encoded_position
        self.state_id = new_state_id
        
    def act(self, t):
        if self.play:
            best_action = np.argmax(self.q[self.position])

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_FOR_QAGENT  * (1 - t / ITERATION_LEARNING):
                best_action = np.random.randint(self.number_actions)
            
            else:
                best_action = np.argmax(self.q[self.position])
            
        return best_action

    def record_reward(self, t):
        """
        save the reward in a file following this pattern:
        iteration_1 reward_1
        iteration_2 reward_2
        iteration_3 reward_3
        """
        self.save_data.record_data(t, self.reward)
        

