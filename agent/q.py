from planning.tree import Tree
import numpy as np


class QAbstract(object):
    """ 
    Contains the q value function which maps a state and an action to a value
    q is a list. Each element contains a structure (list, dictionary etc.) which maps the actions to their values.
    """
    def __str__(self):
        message = ""
        for state in self.get_states():
            for action in self.get_actions(state):
                message += "state " + str(state) + \
                           " action " + str(action) + \
                           " value : " + str(self.get_value(state, action)) + \
                           "\n"

        return message

    def get_states(self):
        """
        returns the state visited so far.
        """
        raise NotImplementedError()

    def get_actions(self, state):
        """
        returns the actions at this state
        """
        raise NotImplementedError()

    def get_value(self, state, action):
        """
        get the value of this state-action pair.
        """
        raise NotImplementedError()

    def set_value(self, state, action, value):
        """
        set this value to this state-action pair.
        """
        raise NotImplementedError()

    def is_state(self, state):
        return state in self.get_states()

    def is_actions(self, state):
        assert self.is_state(state)
        return self.get_actions(state) != []

    def is_action_to_state(self, state, action):
        assert self.is_state(state)
        return action in self.get_actions(state)

    def add_state(self, state):
        raise NotImplementedError()

    def add_action_to_state(self, state, action):
        raise NotImplementedError()

    def find_best_action(self, state):
        """
        output : best_value, best_action
        """
        raise NotImplementedError()

    def update_q_action_state(self, state, new_state, action):
        self.add_action_to_state(state, action)
        self.add_state(new_state)

    def update_q_value(self, state, action, reward, new_state, learning_rate):
        assert self.is_state(state)
        if self.is_actions(new_state):
            best_value, _ = self.find_best_action(new_state)

        else:
            best_value = 0

        self.learning_update(state, action, reward, best_value, learning_rate)

    def learning_update(self, state, action, reward, best_value, learning_rate):
        """
        Q learning procedure :
        Q_{t+1}(current_position, action) =
        (1- learning_rate) * Q_t(current_position, action)
        + learning_rate * [reward + max_{actions} Q_(new_position, action)]
        """
        previous_value = self.get_value(state, action)
        self.set_value(state, action, previous_value * (1 - learning_rate) + learning_rate * (reward + best_value))

    def get_random_action(self, state):
        raise NotImplementedError()

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


class QTree(QAbstract):
    """
    This class is used when the number of actions is unknown.
    state_list = [state_1, state_2, ...] # a list of all states
    each state has a value
    Note that Node.data is a state

    The agent keeps track of the current node with variable current_node *just to color the graph from str_tree*
    """

    def __init__(self, state):
        self.tree = Tree(state)
        self.current_node = self.tree.root

    def __len__(self):
        return len(self.get_states())

    def add_action_to_state(self, state, action):
        pass

    def get_number_options(self):
        nb_children = [len(node.children) for node in self.tree.root.depth_first()]
        return max(nb_children)

    def reset(self):
        self.current_node = self.tree.root
        # self.current_node = self.get_node_from_state(initial_state) # use this for restart agent from a found reward

    def str_tree(self, next_node_data):
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
        node = self.get_node_from_state(state)
        return len(node.children)

    def get_value(self, state, action):
        node = self.get_node_from_state(state)
        return node.children[action]

    def set_value(self, state, action, value):
        node = self.get_node_from_state(state)
        node.children[action] = value

    def add_state(self, state):
        self.current_node = self.tree.add(self.current_node, state)
        self.current_node.data = state

    def get_tree_advices(self):
        return Tree.get_random_next_option_index(self.current_node)

    def find_best_action(self):
        """
        :return: action index, terminal state
        """
        values = self.current_node.get_values()
        best_reward = max(values)
        best_option_index = values.index(best_reward)

        if best_reward == 0:  # In case where there is no best solution: ask the Tree
            best_option_index = self.get_tree_advices()

        return best_option_index, terminal_state

    def get_random_action(self, state):
        return np.random.randint(len(self.current_node.children))

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
            message += str(self.state_list[idx]) + " actions-values : " + str(self.values[idx])

        return message

    def get_actions(self, state):
        """
        No need to implement this function
        """
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
        return self.values[state_idx][action]

    def set_value(self, state, action, value):
        """
        action should be an integer between 0 and number_actions - 1
        """
        state_idx = self.state_list.index(state)
        self.values[state_idx][action] = value

    def add_state(self, state):
        self.state_list.append(state)
        self.values.append(np.zeros(self.number_actions, dtype=np.float64))

    def add_action_to_state(self, state, action):
        assert self.is_state(state)

    def find_best_action(self, state):
        # start = time. time()
        assert self.is_state(state)
        state_idx = self.state_list.index(state)
        # end = time. time()
        # print("time " + str(end - start))
        return np.max(self.values[state_idx]), np.argmax(self.values[state_idx])

    def get_random_action(self, state):
        return np.random.randint(self.number_actions)

    def is_actions(self, state):
        assert self.is_state(state)
        return True

    def is_action_to_state(self, state, action):
        assert self.is_state(state)
        return True

    def update_q_value(self, state, action, reward, new_state, learning_rate, end_option=None):
        new_state_idx = self.state_list.index(new_state)
        state_idx = self.state_list.index(state)
        if end_option:
            best_value = 0

        else:
            best_value = np.max(self.values[new_state_idx])

        self.values[state_idx][action] *= (1 - learning_rate)
        self.values[state_idx][action] += learning_rate * (reward + best_value)
