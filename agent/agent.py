""" This is an abstract class for agents"""

from gridenvs.keyboard_controller import Controls, Key
from gridenvs.utils import Direction, Point
import numpy as np
import time
from agent.option import Option, OptionExplore, OptionExploreQ
from agent.q import Q
from variables import *
from data.save import SaveData

class AgentOption(): 

    def __init__(self, position, state, play, grid_size_option):
        """
        TODO : replace state by a dictionary : self.state = {'zone' : zone, 'state_id' = 0}
        """
        self.last_action = 0
        self.play = play
        self.grid_size_option = grid_size_option
        self.state = state
        self.q = Q(self.state)
        self.position = position
        self.reward = 0
        if not(play):
            self.explore_option = OptionExploreQ(self.position, self.state, self.grid_size_option, 0) # special explore options
#            self.explore_option = OptionExplore(self.state) # special explore options

    def make_save_data(self, seed):
        self.save_data = SaveData("data/options/data_reward_" + self.__class__.__name__, seed)
        
    def reset_explore_option(self):
        self.explore_option.reward = 0
        self.explore_option.initial_state = self.state
        if type(self.explore_option).__name__ == "OptionExploreQ":
            self.explore_option.position = self.explore_option.get_position(self.position)
            self.explore_option.cardinal = self.explore_option.get_cardinal(self.last_action)
            # if this state is explored for the first time : make a new q function full of zeros
            # and set the exploration_terminated boolean for this state to False
            try:
                self.explore_option.q[self.state]
            except:
                self.explore_option.q.update({self.state : np.zeros((self.explore_option.number_state, self.explore_option.number_actions))})
                self.explore_option.exploration_terminated.update({self.state[0] : False})
        
        
    def reset(self, initial_agent_position, initial_agent_state):
        """
        Same as __init__ but the q function is preserved 
        """
        self.reward = 0
        self.position = initial_agent_position
        self.state = initial_agent_state
        self.reset_explore_option()

    def choose_option(self, t):
        """
        if no option : explore
        else flip a coin, then take the best or explore
        """
        if self.play: # in this case we do not learn anymore
            _, best_option = self.q.find_best_action(self.state)
            best_option.play = True
            best_option.position = best_option.get_position(self.position)
            return best_option

        else:
            # No option available : explore, and do not count the number of explorations
            if not(self.q.is_actions(self.state)): 
                self.reset_explore_option()
                return self.explore_option
    
            # options are available : if the exploration is not done then continue exploring
            elif (type(self.explore_option).__name__ == "OptionExploreQ"
                  and not(self.explore_option.exploration_terminated[self.state[0]])):
                self.reset_explore_option()
                return self.explore_option
        
            # in this case find the best option
            else:
                best_reward, best_option = self.q.find_best_action(self.state)
                if best_reward == 0:
                    best_option = np.random.choice(list(self.q.q_dict[self.state].keys()))
                    best_option.position = best_option.get_position(self.position)
                    best_option.reward = 0
                    return best_option
                
                else:
                    best_option.position = best_option.get_position(self.position)
                    best_option.reward = 0
                    return best_option
                        
    def update_agent(self, new_position, new_state, option, action):
        print(self.q)
        if self.play:
            self.state = new_state
            self.position = new_position
            
        else:
            self.last_action = Direction.cardinal().index(action)
            total_reward = self.compute_total_reward(new_state[1])
            self.reward += option.reward_for_agent
            self.update_q_function_options(new_state, option, total_reward)
            self.state = new_state
            self.position = new_position
            
    def update_q_function_options(self, new_state, option, reward):
        """
        only update option(state b, state a) in state b if option(state a, state b) does not already exist in state a.
        """
        if self.no_return_update(new_state):
            assert self.state[0] - new_state[0] in [Point(0, 1), Point(0, 0), Point(0, -1), Point(1, 0), Point(-1, 0)], "options can only jump from a zone to another adjacent one"
            action = Option(self.position, self.state, new_state, self.grid_size_option, self.play)
            # if the state and the action already exist, this line will do nothing
            self.q.update_q_dict_action_space(self.state, new_state, action, reward)
            if option != self.explore_option:
                self.q.update_q_dict_value(self.state, option, reward, new_state)
                
    def no_return_update(self, new_state):
        """
        (no return option)
            does not add anything if 
            for action in q[option.terminal_state]:
            action.terminal_state = option.initial_state
        """
        if self.q.is_state(new_state):
            for action in self.q.q_dict[new_state]:
                if action.terminal_state == self.state:
                    return False

        return True
    
    def compute_total_reward(self, new_state_id):
        total_reward = PENALTY_AGENT_ACTION
        if self.state[1] < new_state_id: # we get an item from the world
            total_reward += REWARD_KEY # extra reward for having the key !

        return total_reward
        
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
    def __init__(self, position, grid_size, play):
        self.play = play
        self.grid_size = grid_size
        self.number_state = 2 * grid_size.x * grid_size.y + 1
        self.number_actions = len(Direction.cardinal())
        self.q = np.zeros((self.number_state, self.number_actions))
        self.cardinal = Direction.cardinal()
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

    def encode_direction(self, direction):
        """
        this function encodes a direction Direction.N/S/E/W into a number, 1/2/3/4
        """
        return self.cardinal.index(direction)
        
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
            
        return self.cardinal[best_action]

    def record_reward(self, t):
        """
        save the reward in a file following this pattern:
        iteration_1 reward_1
        iteration_2 reward_2
        iteration_3 reward_3
        """
        self.save_data.record_data(t, self.reward)
        

