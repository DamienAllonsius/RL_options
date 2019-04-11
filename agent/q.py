import numpy as np
from planning.tree import Node, Tree
from abc import ABCMeta, abstractmethod


class QAbstract(object):
    """
    Contains the q value function which maps a state and an action to a value
    q is a list which elements represent states.
    Each element of q contains a structure (list, dictionary etc.) which maps
    the actions to their values.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def add_state(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def find_best_action(self, state):
        """
        :param state:
        :return: best_value, best_action
        """
        raise NotImplementedError()

    @abstractmethod
    def get_random_action(self, state):
        raise NotImplementedError()

    @abstractmethod
    def update_q_value(self, *args):
        """
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_(new_position, action)]
        :param args:
        :return:
        """
        raise NotImplementedError()


class QTree(QAbstract):
    """
    This class is used when the number of actions is unknown.
    state_list = [state_1, state_2, ...] # a list of all states
    each state has a value
    Note that Node.data is a state
    :param: states are *terminal* state of options
    :param: actions are children index of states
    """
    def __init__(self, state):
        self.tree = Tree(state)
        self.current_node = self.tree.root
        self.number_options = 0

    def __len__(self):
        return len(self.tree.nodes)

    def __str__(self):
        return self.tree.str_tree()

    def reset(self):
        self.current_node = self.tree.root

    def get_node_from_state(self, state):
        """
        :param state:
        :return: the corresponding node with node.data == state
        :exception if the state does not exist
        """
        for node in self.tree.root.depth_first():
            if node.data == state:
                return node

        raise ValueError("state does not exist in the tree")

    def get_child_node_from_current_state(self, state):
        """
        :param state: the node data we are looking for
        :return: a child of self.current_node with child.data == state
        """
        for child in self.current_node.children:
            if child.data == state:
                return child

        raise ValueError("None of my children have this state")

    def add_state(self, next_state):
        """
         Add a state at the current node. But be careful, do not to add twice the same state at the same position.
         :param next_state: the state you want to add
         :return:
         """
        # update the number of visits of the current node
        self.current_node.number_visits += 1
        try:
            self.current_node = self.get_node_from_state(next_state)

        except ValueError: # add next_state only if it does not already exist
            next_current_node = self.tree.add_tree(self.current_node, Node(next_state))
            # and update the number of options
            if len(self.current_node.children) > self.number_options:
                self.number_options += 1

            self.current_node = next_current_node

    def get_random_action(self, state):
        """
        could implement the following code:

        node = self.get_node_from_state(state)
        return np.random.randint(len(node.children))

        but I'm not sure it is worth performing random actions at the high level.
        """
        pass

    def get_number_visits(self):
        return self.current_node.number_visits

    def find_best_action(self, state=None):
        """
        :return: best_option_index, terminal_state
        """
        values = self.current_node.get_values()
        if not values:
            return 0, None

        # In case where there is no best solution: ask the Tree
        if all(val == values[0] for val in values):
            best_option_index = Tree.get_random_next_option_index(self.current_node)

        else:
            best_reward = max(values)
            best_option_index = values.index(best_reward)

        return best_option_index, self.current_node.children[best_option_index].data

    def update_q_value(self, action, reward, new_state, learning_rate):
        """
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_(new_position, action)]

        """
        node_activated = self.get_child_node_from_current_state(action)  # node which value attribute is
        # Q_t(current_position, action)

        try:
            new_node = self.get_node_from_state(new_state)  # maybe different than node_activated
            if new_node.children:  # there are children, take the maximum value
                best_value = max(new_node.get_values())

            else:  # there are no children -> best_value is 0
                best_value = 0

        except ValueError:  # this new_state does not exist for the moment
            best_value = 0

        node_activated.value *= (1 - learning_rate)
        node_activated.value += learning_rate * (reward + best_value)

    # def no_return_update(self, state, new_state):
    #     """
    #     (no return option)
    #         does not add anything if
    #         for action in q[option.terminal_state]:
    #         action.terminal_state = option.initial_state
    #     """
    #     if self.is_state(new_state):
    #         for node in self.get_actions(new_state):
    #             if node.data == state:
    #                 return False
    #
    #     return True


class QArray(QAbstract):
    """
    _ This class is used when the number of states is unknown but the number of actions is known
    _ state_list = [state_1, state_2, ...] # a list of all states
    _ Unlike QTree, this class does not allow an efficient exploration.
    _ But here the Q function cannot be represented with a Tree
      because there exists set of states which are not connected.
      This about the transition from a zone to another, the agent may be in two different position at the entrance of
      a new zone.
    _Action should be an integer between 0 and number_actions - 1
    """
    def __init__(self, state, number_actions):
        self.state_list = [state]
        self.values = [np.zeros(number_actions, dtype=np.float64)]
        self.number_actions = number_actions

    def __len__(self):
        """
        :return: number of states
        """
        return len(self.state_list)

    def __str__(self):
        message = ""
        for idx in range(len(self.state_list)):
            message += str(self.state_list[idx]) + \
                       " actions-values : " + str(self.values[idx]) + \
                       "\n"

        return message

    def add_state(self, next_state):
        self.state_list.append(next_state)
        self.values.append(np.zeros(self.number_actions, dtype=np.float64))

    def find_best_action(self, state):
        """
        :param state:
        :return: best_action
        """
        assert state in self.state_list
        state_idx = self.state_list.index(state)

        return np.argmax(self.values[state_idx])

    def get_random_action(self, state):
        return np.random.randint(self.number_actions)

    def update_q_value(self, state, action, reward, new_state, end_option, learning_rate):
        new_state_idx = self.state_list.index(new_state)
        state_idx = self.state_list.index(state)
        if end_option:
            best_value = 0

        else:
            best_value = np.max(self.values[new_state_idx])

        self.values[state_idx][action] *= (1 - learning_rate)
        self.values[state_idx][action] += learning_rate * (reward + best_value)
