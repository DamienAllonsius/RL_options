""" This is an abstract class for agents"""

from gridenvs.keyboard_controller import Controls, Key
from gridenvs.utils import Point
import numpy as np
import time

from agent.option import Option, OptionExplore#, OptionExploreQ
from agent.q import QTree
from variables import *
from data.save import SaveData

class AgentOption(object): 

    def __init__(self, current_state, number_actions, type_exploration, play):
        self.number_actions = 8#number_actions
#        self.last_action = 0 #last action : north, east, south, west ?
        self.play = play
        self.current_state = current_state
        self.q = QTree(current_state["blurred_state"])
        self.personal_reward = 0 # the personal reward of the agent, not used for updating policy
        self.type_exploration = type_exploration
        if not(play):
            if type_exploration == "OptionExplore":
                self.explore_option = OptionExplore(current_state["blurred_state"], self.number_actions) # special explore options

            elif type_exploration == "OptionExploreQ":
                raise Exception("Fix it !")
                self.explore_option = OptionExploreQ(current_state, self.number_actions) # special explore options
                #before on IW branch: self.explore_option = OptionExploreQ(self.position, self.state, self.grid_size_option) # special explore options
                
            else:
                raise Exception("type_exploration unknown")

    # def make_save_data(self, seed):
    #     #TODO : monitor ?
    #     self.save_data = SaveData("data/options/data_reward_" + self.__class__.__name__, seed)

    def display_Qtree(self, next_node_data):
        """
        for QTree
        """
        print(self.q.str_QTree(next_node_data))
        
    # def display_QDict(self, option):
    #     """
    #     for QDict
    #     """
    #     message = ""
    #     for state_index in range(len(self.q.q_function)):
    #         for action in self.q.q_function[state_index]:
    #             txt = "state " + str(self.q.state_list[state_index]) + " action " + str(action) + " value : " + str(self.q.q_function[state_index][action]) + "\n"
    #             if action == option:
    #                 message += '\033[92m' + txt + '\033[0m'

    #             else:
    #                 message += txt
                
    #     return message

    def reset_explore_option(self):
        self.explore_option.reward_for_agent = 0
        self.explore_option.initial_state = self.current_state["blurred_state"]
        """
        TODO
        """
         # if type(self.explore_option).__name__ == "OptionExploreQ":
         #    self.explore_option.position = self.explore_option.get_position(self.position)
         #    #self.explore_option.cardinal = self.explore_option.get_cardinal(self.last_action)
         #    # if this state is explored for the first time : make a new q function full of zeros
         #    # and set the exploration_terminated boolean for this state to False
         #    try:
         #        self.explore_option.q[self.state]
         #    except:
         #        self.explore_option.q.update({self.state : np.zeros((self.explore_option.number_state, self.explore_option.number_actions))})
         #        self.explore_option.exploration_terminated.update({self.state : False})

        
    def reset(self, initial_agent_state):
        """
        Same as __init__ but the q function is preserved
        """
        self.q.reset()
        self.personal_reward = 0
        self.current_state = initial_agent_state
        self.reset_explore_option()

    def choose_option(self, t):
        """
        if no option : explore
        else flip a coin, then take the best or explore
        """
        if self.play: # in this case we do not learn anymore
            _, best_option = self.q.find_best_action(self.current_state["blurred_state"])
            best_option.play = True
            return best_option

        else:
            # No option available : explore, and do not count the number of explorations
            if not(self.q.is_actions(self.current_state["blurred_state"])):
                self.reset_explore_option()
                return self.explore_option
            
            # options are available : if the exploration is not done then continue exploring

            elif (self.type_exploration == "OptionExplore" and
                  not(self.explore_option.check_end_option(self.current_state["blurred_state"]))):
                self.reset_explore_option()
                return self.explore_option

            elif (type(self.explore_option).__name__ == "OptionExploreQ"
                  and not(self.explore_option.exploration_terminated[self.current_state["blurred_state"]])):
                self.reset_explore_option()
                return self.explore_option

            else:
                best_reward, best_option = self.q.find_best_action(self.current_state["blurred_state"])
                if best_reward == 0:
                    next_terminal_state = self.q.get_tree_advices()
                    for opt in self.q.get_actions(self.current_state["blurred_state"]):
                        if opt.terminal_state == next_terminal_state:
                            best_option = opt
                    
                best_option.reward_for_agent = 0
                best_option.set_current_state(self.current_state["state"])  # update the option's current state ! 
                #print("0. agent.q " + str(self.q))
                #print("1. best option : " +str(best_option))
                #print("2. best_option.q : " +str(best_option.q))
                return best_option
            
    def update_agent(self, new_state, option, action):
        """
        In this order
        _update last action done
        _update reward
        _add an option if a new state has just been discovered
        _update the q function value
        _update the state
        """
        #self.display_Qtree(new_state["blurred_state"])
        if self.play:
            self.current_state = new_state
            
        else:
#            self.last_action = action
            if option.reward_for_agent > 0: # the worker found a positive reward
                import subprocess
                subprocess.Popen(['notify-send', "got a posive reward !"])
                print("\033[93m got a posive reward !")
                
            total_reward = PENALTY_AGENT_ACTION + option.reward_for_agent 
            self.update_q_function_options(new_state, option, total_reward)
            self.current_state = new_state
            self.personal_reward += option.reward_for_agent
            
    def update_q_function_options(self, new_state, option, reward):
        if self.q.no_return_update(self.current_state["blurred_state"], new_state["blurred_state"]):
            
            action = Option(self.number_actions, self.current_state["blurred_state"], new_state["state"], new_state["blurred_state"], self.play)
            
            # if the state and the action already exist, this line will do nothing
            self.q.update_q_action_state(self.current_state["blurred_state"], new_state["blurred_state"], action)
            print("number options " + str(len(self.q)))
            if option != self.explore_option:
                self.q.update_q_value(self.current_state["blurred_state"], option, reward, new_state["blurred_state"])

        else:
            self.q.update_current_node(new_state["blurred_state"])
    
    # def record_reward(self, t):
    #     """
    #     save the reward in a file following this pattern:
    #     iteration_1 reward_1
    #     iteration_2 reward_2
    #     iteration_3 reward_3
    #     """
    #     self.save_data.record_data(t, self.personal_reward)


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
    """
    TODO
    this class has to be refactored 
    """
    def __init__(self, position, number_actions, grid_size, play):
        self.play = play
        self.grid_size = grid_size
        self.number_state = 2 * grid_size.x * grid_size.y + 1
        self.number_actions = number_actions
        self.q = np.zeros((self.number_state, self.number_actions))
        self.state_id = 0
        self.position = self.encode_position(position)
        self.personal_reward = 0

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
        self.personal_reward = 0
        self.position = initial_position

    def update(self, reward, new_position, action, new_state_id):
        encoded_action = self.encode_direction(action)
        encoded_position = self.encode_position(new_position)
        max_value_action = np.max(self.q[encoded_position])
        total_reward = reward + PENALTY_ACTION
        self.personal_reward += total_reward
        self.q[self.position, encoded_action] *= (1 - LEARNING_RATE)
        self.q[self.position, encoded_action] += LEARNING_RATE * (total_reward + max_value_action)
        self.position = encoded_position
        self.state_id = new_state_id
        
    def act(self, t):
        if self.play:
            best_action = np.argmax(self.q[self.position])

        else:
            if np.random.rand() < PROBABILITY_EXPLORE_FOR_QAGENT * (1 - t / ITERATION_LEARNING):
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
        self.save_data.record_data(t, self.personal_reward)
        

