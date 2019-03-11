from gridenvs.utils import Point, Direction
from variables import *
from planning.tree import Tree
import numpy as np
import time

class QAbstract(object):
    """ 
    Contains the q value function which maps a state and an action to a value
    q is a list. Each element contains a structure (list, dictionary etc.) which maps the actions to their values.
    """
    def __str__(self):
        message = ""
        for state in self.get_states():
            for action in self.get_actions(state):
                message += "state " + str(state) + " action " + str(action) + " value : " + str(self.get_value(state, action)) + "\n"
                
        return message

    def __len__(self):
       return len(self.get_states)

    def get_states(self):
        """
        returns the state visited so far.
        """
        raise Exception("Not Implemented")

    def get_actions(self, state):
        """
        returns the actions at this state
        """
        raise Exception("Not Implemented")

    def get_value(self, state, action):
        """
        get the value of this state-action pair.
        """
        raise Exception("Not Implemented")
   
    def set_value(self, state, action, value):
        """
        set this value to this state-action pair.
        """
        raise Exception("Not Implemented")

    def is_state(self, state):
        return state in self.get_states()
    
    def is_actions(self, state):
        if not(self.is_state(state)):
            raise Exception("cannot test if actions exist since state does not exist")
        
        else:
            return self.get_actions(state) != []
   
    def add_state(self, state):
        raise Exception("Not Implemented")

    def add_action_to_state(self, state, action):
        raise Exception("Not Implemented")
        
    def find_best_action(self, state):
        """
        output : best_value, best_action
        """
        raise Exception("Not Implemented")

    def update_q_action_state(self, state, new_state, action):
        """
        was previously: add action to state
        """
        self.add_action_to_state(state, action)
        self.add_state(new_state)

    def update_q_value(self, state, action, reward, new_state):
        if self.is_state(state):
            if self.is_actions(new_state):
                best_value, _ = self.find_best_action(new_state)

            else:
                best_value = 0

            self.learning_update(self, state, action, reward, best_value)

        else:
            raise Exception('unhable to update q since state does not exist')

    def learning_update(self, state, action, reward, best_value):
        """
        Q learning procedure :
        Q_{t+1}(current_position, action) =
        (1- learning_rate) * Q_t(current_position, action)
        + learning_rate * [reward + max_{actions} Q_(new_position, action)]
        """
        previous_value = self.get_value(state, action)
        self.set_value(state, action, previous_value * (1 - LEARNING_RATE) + LEARNING_RATE * (reward + best_value))

    def get_random_action(self, state):
        raise Exception("Not Implemented")

    def no_return_update(self, state, new_state):
        """
        (no return option)
            does not add anything if 
            for action in q[option.terminal_state]:
            action.terminal_state = option.initial_state
        """
        if self.is_state(new_state):
            for option in self.get_actions(new_state):
                if option.terminal_state == state:
                    return False

        return True

class QArray(QAbstract):
    """
    this class is used when the number of actions known.
    states are integers representing the positions of the agent
    """
    def __init__(self, number_states, number_actions):
        self.q_function = np.zeros((number_states, number_actions), dtype = np.float64)
        self.number_actions = number_actions
        self.number_states = number_states

    def __str__(self):
        return str(self.q_function)

    def find_best_action(self, state):
        return np.max(self.q_function[state]), np.argmax(self.q_function[state])

    def get_states(self):
        return range(self.number_states)

    def get_actions(self, state):
        return range(self.number_actions)
  
    def get_value(self, state, action):
        return self.q_function[state, action]

    def get_random_action(self, state):
        return np.random.randint(self.number_actions)

    def update_q_value(self, state, action, reward, new_state, end_option):
        if end_option:
            best_value = 0

        else:
            best_value = np.max(self.q_function[new_state])

        self.q_function[state, action] = (1 - LEARNING_RATE) * self.q_function[state, action]  + LEARNING_RATE * (reward + best_value)
        
class QList(QAbstract):
    """
    This class is used when the number of actions is unknown.
    state_list = [state_1, state_2, ...] # a list of all states
    actions = [[action_1_state_1, action_2_state_1], [action_1_state_2], ...]
    values = [[value_action_1_state_1, value_action_2_state_1], [value_action_1_state_2], ...]
    Unlike QTree, this class does not allow an efficient exploration.
    """
    def __init__(self, state):
        self.state_list = [state]
        self.actions = [[]]
        self.values = [[]]

    def get_states(self):
        """
        returns the state visited so far.
        """
        return self.state_list

    def get_actions(self, state):
        state_idx = self.state_list.index(state)
        return self.actions[state_idx]
    
    def get_value(self, state, action):
        state_idx = self.state_list.index(state)
        action_idx = self.actions[state_idx].index(action)
        return self.values[state_idx][action_idx]
    
    def set_value(self, state, action, value):
        state_idx = self.state_list.index(state)
        action_idx = self.actions[state_idx].index(action)
        self.values[state_idx][action_idx] = value
    
    def add_state(self, state):
        self.state_list.append(state)
        self.actions.append([])
        self.values.append([])
        
    def add_action_to_state(self, state, action):
        if not(self.is_state(state)):
            raise Exception("action cannot be added since state does not exist")

        else:
            state_idx = self.state_list.index(state)
            self.actions[state_idx].append(action)
            self.values[state_idx].append(0)
        
    def find_best_action(self, state):
        #start = time. time()
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        elif not(self.is_actions(state)):
            raise Exception('cannot find best action since there is no action in state ' + str(state))

        else: # return best_value, best_action
            state_idx = self.state_list.index(state)
            m = max(self.values[state_idx])
            #end = time. time()
            #print("time " + str(end - start))
            return m, self.actions[state_idx][self.values[state_idx].index(m)]

    def get_random_action(self, state):
        return np.random.choice(self.actions[self.state_list.index(state)])

class QTree(QAbstract):
    """
    This class is used when the number of actions is unknown.
    state_list = [state_1, state_2, ...] # a list of all states
    actions = [[action_1_state_1, action_2_state_1], [action_1_state_2], ...]
    values = [[value_action_1_state_1, value_action_2_state_1], [value_action_1_state_2], ...]
    Unlike QTree, this class does not allow an efficient exploration.

    Node.data = state
    """
    def __init__(self, state):
        self.tree = Tree(state)
        self.current_node = self.tree.root
        
    def get_states(self):
        """
        returns the state visited so far.
        """
        return self.tree.nodes

    def get_node_from_state(self, state):
        for node in self.tree.root.depth_first():
            if node.state == state:
                return node

        raise Exception("state does not exist in the tree")

    def get_actions(self, state):
        """
        the agent should keep track of the current node
        """
        node = self.get_node_from_state(state)
        return node.actions

    def get_value(self, state, action):
        node = self.get_node_from_state(state)
        idx = node.actions.index(action)
        return node.values[idx]
        
    def set_value(self, state, action, value):
        node = self.get_node_from_state(state)
        idx = node.actions.index(action)
        node.values[idx] = value
        
    def add_state(self, state):
        self.current_node = self.tree.add(self.current_node, node)
        self.current_node.state = state
        n = self.get_node_from_state(state) # just a test, exception should not be raised
        
    def add_action_to_state(self, state, action):
        self.current_node.actions.append(action)
            
    def find_best_action(self, state):
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        elif not(self.is_actions(state)):
            raise Exception('cannot find best action since there is no action in state ' + str(state))

        else: # return best_value, best_action
            m = max(self.current_node.values)
            return m, self.actions.index(m)

    def get_random_action(self, state):
        return np.random.choice(self.current_node.actions)
