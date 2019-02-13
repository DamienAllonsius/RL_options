from gridenvs.utils import Point, Direction
from variables import *

class QAbstract(object):
    """ 
    Contains the q value function which maps a state and an action to a value
    q is a list of dictionaries q[n] = {action_1 : value_1, action_2 : value_2}
    TODO : find a way to print the states. Maybe print the screen ? 
    """
    def __init__(self, state):
        self.state_list = [state]
        self.q = [{}]
        self.number_states = 1
        
    def __str__(self):
        message = ""
        for state_index in self.q:
            for action in self.q[state_index]:
                message += "state " +str(self.state_list[state_index]) + " action " + str(action) + " value : " + str(self.q_dict[state_index][action]) + "\n"
                
        return message

    def is_state(self, state):
        return state in self.state_list

    def is_actions(self, state):
        if not(self.is_state(state)):
            raise Exception("cannot test if actions exist since state does not exist")
        
        else:
            return self.q[self.state_list.index(state)] != {}

    def is_action_to_state(self, state, action):
        if self.is_state(state) and self.is_actions(state):
            return action in self.q_dict[self.state_list.index(state)]
        
    def add_state(self, state):
        """
        If state does not exist, we create it.
        Otherwise, we do nothing.
        """
        if not(self.is_state(state)):
            self.state_list.append(state)
            self.q.append({})
            self.number_states += 1
        
    def add_action_to_state(self, state, action, reward = 0):
        """
        ? idea ? :
        does not add anything if 
        for action in q[option.terminal_state]:
           action.terminal_state = option.initial_state
        """
        if not(self.is_state(state)):
            raise Exception("action cannot be added since state does not exist")

        else:
            self.q[self.state_list.index(state)].update({action : reward})
        
    def find_best_action(self, state):
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        elif not(self.is_actions(state)):
            raise Exception('cannot find best action since there is no action in state ' + str(state))

        else: # return best_value, best_action
            idx = self.state_list.index(state)
            values = list(self.q[idx].values())
            actions = list(self.q[idx].keys())
            return max(values), actions[values.index(max(values))]    
 
    def update_q_dict_action_space(self, state, new_state, action, reward):
        self.add_action_to_state(state, action, reward)
        self.add_state(new_state)

    def update_q_dict_value(self, state, action, reward, new_state):
        raise Exception ("Not implemented")

class Q(QAbstract):

    def update_q_dict_value(self, state, action, reward, new_state):
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
            self.q[idx][action] *= (1 - LEARNING_RATE)
            self.q[idx][action] += LEARNING_RATE * (reward + best_value)

        else:
            raise Exception('unhable to update q since state does not exist')

