import numpy as np
from planning.tree import Node, Tree


class QAbstract(object):
    """
    Contains the q value function which maps a state and an action to a value
    q is a list which elements represent states.
    Each element of q contains a structure (list, dictionary etc.) which maps
    the actions to their values.
    """
    def add_state(self, *args):
        raise NotImplementedError()

    def find_best_action(self, state):
        """
        :param state:
        :return: best_value, best_action
        """
        raise NotImplementedError()

    def get_random_action(self, state):
        raise NotImplementedError()

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

    def __len__(self):
        return len(self.tree.nodes)

    def __str__(self):
        return self.tree.str_tree()

    def get_node_from_state(self, state):
        """
        :param state:
        :return: the corresponding node with node.data == state
        :exception if the state does not exist
        """
        for node in self.tree.root.depth_first():
            if node.data == state:
                return node

        raise Exception("state does not exist in the tree")

    def get_number_actions(self, state):
        node = self.get_node_from_state(state)
        return len(node.children)

    def add_state(self, previous_state, next_state):
        """
         Add a state at a certain position. But careful not to add twice the same state at the same position.
         :param next_state: the state you want to add
         :param previous_state: the state *before* next_state
         :return:
         """
        node = self.get_node_from_state(previous_state)
        if next_state not in [child.data for child in node.children]:  # add only if the state does not already exist
            self.tree.add_tree(node, Node(next_state))

    def get_random_action(self, state):
        """
        could implement the following code:

        node = self.get_node_from_state(state)
        return np.random.randint(len(node.children))

        but I'm not sure it is worth performing random actions at the high level.
        """
        pass

    def get_number_visits(self, state):
        return self.get_node_from_state(state).number_visits

    def get_number_options(self):
        """
        This function is used only for the following test:
        Must be equal to len(agent.option_list)
        """
        nb_children = [len(node.children) for node in self.tree.root.depth_first()]
        return max(nb_children)

    def find_best_action(self, state):
        """
        :return: best_option_index, terminal_state
        """
        node = self.get_node_from_state(state)
        values = QTree.get_values(node)

        # In case where there is no best solution: ask the Tree
        if all(val == values[0] for val in values):
            best_option_index = Tree.get_random_next_option_index(node)

        else:
            best_reward = max(values)
            best_option_index = values.index(best_reward)

        return best_option_index, node.children[best_option_index].data

    def update_q_value(self, state, action, reward, new_state, learning_rate):
        """
        Performs the Q learning update :
        Q_{t+1}(current_position, action) = (1- learning_rate) * Q_t(current_position, action)
                                         += learning_rate * [reward + max_{actions} Q_(new_position, action)]
        """
        node = self.get_node_from_state(state)
        node_activated = node[node.index(action)]  # node which value attribute is Q_t(current_position, action)
        new_node = self.get_node_from_state(new_state)  # maybe different than node_activated

        if new_node.children:  # there are children, take the maximum value
            best_value = max(QTree.get_values(new_node))

        else:  # there are no children -> best_value is 0
            best_value = 0

        node_activated.value *= (1 - learning_rate)
        node_activated.value += learning_rate * (reward + best_value)

    @staticmethod
    def get_values(node):
        return [child.value for child in node.children]

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
                       " actions-values : " + str(self.values[idx])

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

    def update_q_value(self, state, action, reward, new_state, learning_rate, end_option):
        new_state_idx = self.state_list.index(new_state)
        state_idx = self.state_list.index(state)
        if end_option:
            best_value = 0

        else:
            best_value = np.max(self.values[new_state_idx])

        self.values[state_idx][action] *= (1 - learning_rate)
        self.values[state_idx][action] += learning_rate * (reward + best_value)
