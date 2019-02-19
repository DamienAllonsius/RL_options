from gridenvs.utils import Point, Direction
from variables import *
import numpy as np

class QAbstract(object):
    """ 
    Contains the q value function which maps a state and an action to a value.
    q is a list. Each element contains a structure (list, dictionary etc.) which maps the actions to their values.
    TODO : find a way to print the states. Maybe print the screen ? 
    """
    def __init__(self, state):
        self.state_list = [state]
        self.q_function = [self.get_empty_structure()]

    def get_empty_structure(self):
        raise Exception("Not Implemented")

    def is_empty(self, struc):
        raise Exception("Not Implemented")
        
    def __str__(self):
        message = ""
        for state_index in range(len(self.q_function)):
            for action in self.q_function[state_index]:
                message += "state " +str(hash(self.state_list[state_index])) + " action " + str(action) + " value : " + str(self.q_function[state_index][action]) + "\n"
                
        return message

    def is_state(self, state):
        return state in self.state_list

    def is_actions(self, state):
        if not(self.is_state(state)):
            raise Exception("cannot test if actions exist since state does not exist")
        
        else:
            return self.is_empty(self.q_function[self.state_list.index(state)])

    def is_action_to_state(self, state, action):
        if self.is_state(state):
            return action in self.q_function[self.state_list.index(state)]
        
        return False
    
    def add_state(self, state):
        if not(self.is_state(state)):
            self.state_list.append(state)
            self.q_function.append(self.get_empty_structure())
        
    def add_action_to_state(self, state, action):
        raise Exception("Not Implemented")
        
    def find_best_action(self, state):
        raise Exception("Not Implemented")

    def update_q_function_action_state(self, state, new_state, action):
        self.add_action_to_state(state, action)
        self.add_state(new_state)

    def update_q_function_value(self, state, action, reward, new_state):
        """
        Q learning procedure :
        Q_{t+1}(current_position, action) =
        (1- learning_rate) * Q_t(current_position, action)
        + learning_rate * [reward + max_{actions} Q_(new_position, action)]
        """
        if self.is_state(state):
            if self.is_actions(new_state):
                best_value, _ = self.find_best_action(new_state)

            else:
                best_value = 0

            idx = self.state_list.index(state)
            self.q_function[idx][action] *= (1 - LEARNING_RATE)
            self.q_function[idx][action] += LEARNING_RATE * (reward + best_value)

        else:
            raise Exception('unhable to update q since state does not exist')

    def get_random_action(self, state):
        raise Exception("Not Implemented")


class QDict(QAbstract):
    """
    this class is used when the number of actions is unknown : the elements of self.q_function are dictionaries.
    """
    def get_empty_structure(self):
        return {}

    def add_action_to_state(self, state, action):
        if not(self.is_state(state)):
            raise Exception("action cannot be added since state does not exist")

        else:
            self.q_function[self.state_list.index(state)].update({action : 0}) 
        
    def find_best_action(self, state):
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        elif not(self.is_actions(state)):
            raise Exception('cannot find best action since there is no action in state ' + str(state))

        else: # return best_value, best_action
            idx = self.state_list.index(state)
            values = list(self.q_function[idx].values())
            actions = list(self.q_function[idx].keys())
            return max(values), actions[values.index(max(values))]

    def get_random_action(self, state):
        state_keys = self.q_function[self.state_list.index(state)].keys()
        return np.random.choice(list(state_keys))

    def is_empty(self, struct):
         return struct != self.get_empty_structure()
 
        
class QArray(QAbstract):
    """
    TODO : is action ?!
    this class is used when the number of actions known : the elements of self.q_function are fixed size arrays.
    """
    def __init__(self, state, number_actions):
        self.number_actions = number_actions
        super().__init__(state)
        
    def get_empty_structure(self):
        return np.zeros(self.number_actions)

    def add_action_to_state(self, state, action):
        pass
        
    def find_best_action(self, state):
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        else: # return best_value, best_action
            idx = self.state_list.index(state)
            return max(self.q_function[idx]), np.argmax(self.q_function[idx])
    
    def is_actions(self, state):
        return True
    
    def is_action_to_state(self, state, action):
        return True

    def get_random_action(self, state):
        """
        TODO
        """
        pass
