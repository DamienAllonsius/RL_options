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

    def reset(self):
        """
        returns the state visited so far.
        """
        raise Exception("Not Implemented")
    
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

    def is_action_to_state(self, state, action):
        if not(self.is_state(state)):
            raise Exception("cannot test if action exist to state since state does not exist")
        
        else:
            return (action in self.get_actions(state))
   
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

            self.learning_update(state, action, reward, best_value)

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

class QList(QAbstract):
    """
    This class is used when the number of states and actions are unknown.
    state_list = [state_1, state_2, ...] # a list of all states
    actions = [[action_1_state_1, action_2_state_1], [action_1_state_2], ...]
    values = [[value_action_1_state_1, value_action_2_state_1], [value_action_1_state_2], ...]
    Unlike QTree, this class does not allow an efficient exploration.
    """
    def __init__(self, state):
        self.state_list = [state]
        self.actions = [[]]
        self.values = [[]]

    def reset(self):
        pass

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
        self.reset()

    def __len__(self):
        nb_opt = 0
        for state in self.get_states():
            nb_opt += len(self.get_actions(state))

        return nb_opt

    def reset(self):
        self.current_node = self.tree.root

    def str_QTree(self, next_node_data):
        return self.tree.str_tree(self.current_node.data, next_node_data)
        
    def get_states(self):
        """
        returns the state visited so far.
        """
        return [node.data for node in self.tree.nodes]

    def get_node_from_state(self, state):
        for node in self.tree.root.depth_first():
            if node.data == state:
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
        self.current_node = self.tree.add(self.current_node, state)
        self.current_node.data = state
        
    def add_action_to_state(self, state, action):
        if not self.is_action_to_state(state, action):
            self.current_node.actions.append(action)
            self.current_node.values.append(0)

    def get_tree_advices(self):
        return self.tree.get_random_next_data(self.current_node)
            
    def find_best_action(self, state):
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        elif not(self.is_actions(state)):
            raise Exception('cannot find best action since there is no action in state ' + str(state))

        else: # return best_value, best_action
            node = self.get_node_from_state(state)
            m = max(node.values)
            return m, node.actions[node.values.index(m)]

    def get_random_action(self, state):
        return np.random.choice(self.current_node.actions)

    def update_current_node(self, new_state):
        self.current_node = self.get_node_from_state(new_state)


class QArray(QAbstract):
    """
    This class is used when the number of states is unknown but the number of actions is known
    state_list = [state_1, state_2, ...] # a list of all states
    Unlike QTree, this class does not allow an efficient exploration.
    """
    def __init__(self, state, number_actions):
        self.state_list = [state]
        self.actions = [np.zeros(number_actions, dtype = np.float64)]
        self.number_actions = number_actions

    def __len__(self):
        l = 0
        for idx in range(len(self.state_list)):
            l += len(self.actions[idx])
    
        return l

    def __str__(self):
        message = ""
        for idx in range(len(state_list)):
            message += str(self.state_list[idx]) + " actions-values : " + str(self.actions[idx])

        return message
     
    def reset(self):
        pass

    def get_states(self):
        """
        returns the state visited so far.
        """
        return self.state_list
    
    def get_value(self, state, action):
        """
        action should be an integer between 0 and number_actions - 1
        """
        state_idx = self.state_list.index(state)
        return self.actions[state_idx][action]
    
    def set_value(self, state, action, value):
        """
        action should be an integer between 0 and number_actions - 1
        """
        state_idx = self.state_list.index(state)
        self.values[state_idx][action] = value
    
    def add_state(self, state):
        self.state_list.append(state)
        self.actions.append(np.zeros(self.number_actions, dtype = np.float64))
        
    def add_action_to_state(self, state, action):
        if not(self.is_state(state)):
            raise Exception("action cannot be added since state does not exist")
        
    def find_best_action(self, state):
        #start = time. time()
        if not(self.is_state(state)):
            raise Exception('cannot find best action since there is no state : ' + str(state))
        
        else: # return best_value, best_action
            state_idx = self.state_list.index(state)
            #end = time. time()
            #print("time " + str(end - start))
            return np.max(self.actions[state_idx]), np.argmax(self.actions[state_idx])

    def get_random_action(self, state):
        return np.random.randint(self.number_actions)

    def is_actions(self, state):
        if not(self.is_state(state)):
            raise Exception("cannot test if actions exist since state does not exist")

        else:
            return True
    
    def is_action_to_state(self, state, action):
        if not(self.is_state(state)):
            raise Exception("cannot test if actions exist since state does not exist")

        else:
            return True

    def update_q_value(self, state, action, reward, new_state, end_option):
        new_state_idx = self.state_list.index(new_state)
        state_idx = self.state_list.index(state)
        if end_option:
            best_value = 0
            
        else:
            best_value = np.max(self.actions[new_state_idx])

        self.actions[state_idx][action] = (1 - LEARNING_RATE) * self.actions[state_idx][action]  + LEARNING_RATE * (reward + best_value)

# class QArray(QAbstract):
#     """
#     TODO : is action ?!
#     this class is used when the number of actions is known : the elements of self.q_function are fixed size arrays.
#     However the number of states is unknown
#     """
#     def __init__(self, state, number_actions):
#         self.number_actions = number_actions
#         super().__init__(state)
        
#     def get_empty_structure(self):
#         return np.zeros(self.number_actions)

#     def add_action_to_state(self, state, action):
#         pass
        
#     def find_best_action(self, state):
#         if not(self.is_state(state)):
#             raise Exception('cannot find best action since there is no state : ' + str(state))
        
#         else: # return best_value, best_action
#             idx = self.state_list.index(state)
#             return max(self.q_function[idx]), np.argmax(self.q_function[idx])
    
#     def is_actions(self, state):
#         return True
    
#     def is_action_to_state(self, state, action):
#         return True

#     def get_random_action(self, state):
#         """
#         TODO
#         """
#         pass

# class QDict(QAbstract):
#     """
#     this class is used when the number of actions is unknown : the elements of self.q_function are dictionaries.
#     """
#     def __len__(self):
#         l = 0
#         for dict_opt in self.q_function:
#             l += len(dict_opt)

#         return l
        
#     def get_empty_structure(self):
#         return {}
    
#     def add_action_to_state(self, state, action):
#         if not(self.is_state(state)):
#             raise Exception("action cannot be added since state does not exist")

#         else:
#             self.q_function[self.state_list.index(state)].update({action : 0}) 
        
#     def find_best_action(self, state):
#         if not(self.is_state(state)):
#             raise Exception('cannot find best action since there is no state : ' + str(state))
        
#         elif not(self.is_actions(state)):
#             raise Exception('cannot find best action since there is no action in state ' + str(state))

#         else: # return best_value, best_action
#             idx = self.state_list.index(state)
#             values = list(self.q_function[idx].values())
#             actions = list(self.q_function[idx].keys())
#             return max(values), actions[values.index(max(values))]

#     def get_random_action(self, state):
#         state_keys = self.q_function[self.state_list.index(state)].keys()
#         return np.random.choice(list(state_keys))

#     def is_empty(self, struct):
#          return struct != self.get_empty_structure()


# class QArray(QAbstract):
#     """
#     this class is used when the number of actions is known but the number of states is unknown.
#     states are integers representing the positions of the agent
#     from IW initial set up.
#     """
#     def __init__(self, state, number_actions):
#         self.state_list = [state]
#         self.q_function = np.zeros((number_states, number_actions), dtype = np.float64)
#         self.number_actions = number_actions
#         self.number_states = number_states

#     def __str__(self):
#         return str(self.q_function)

#     def reset(self):
#         pass
    
#     def find_best_action(self, state):
#         return np.max(self.q_function[state]), np.argmax(self.q_function[state])

#     def get_states(self):
#         return range(self.number_states)

#     def get_actions(self, state):
#         return range(self.number_actions)
  
#     def get_value(self, state, action):
#         return self.q_function[state, action]

#     def get_random_action(self, state):
#         return np.random.randint(self.number_actions)

#     def update_q_value(self, state, action, reward, new_state, end_option):
#         if end_option:
#             best_value = 0

#         else:
#             best_value = np.max(self.q_function[new_state])

#         self.q_function[state, action] = (1 - LEARNING_RATE) * self.q_function[state, action]  + LEARNING_RATE * (reward + best_value)
