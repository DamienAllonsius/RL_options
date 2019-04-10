from collections import defaultdict
from planning.utils import *
import numpy as np
import variables


class Node(object):
    def __init__(self, data, parent=None):
        self.value = 0
        self.number_visits = 0
        self.data = data  # a.k.a state

        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1

        else:
            self.depth = 0

        self.children = []

    def __repr__(self):
        return "data: " + str(self.data)

    def __str__(self):
        s = str(self.data) + " at depth " + str(self.depth) + "\n"
        if self.parent is not None:
            s += "parent: " + str(self.parent.data) + "\n"

        s += "node and children:" + "\n"
        s += self.str_node(self)
        for k in range(len(self.children)):
            s += "action: " + str(k) + \
                 " value: " + str(self.children[k]) \
                 + "\n"

        return s

    def str_node(self,
                 current_node=None,
                 next_node=None,
                 str_data_fn=lambda node: str(node.data)):

        if self.data == current_node:
            s = variables.red

        else:
            s = variables.green

        s += str_data_fn(self) + '\n'
        for node in self.depth_first():
            d = node.depth - self.depth
            if d > 0:
                tex = "".join([variables.tab] * d +
                              ["|", str_data_fn(node), '\n'])
                if node.data == current_node:
                    s += variables.red

                elif node.data == next_node:
                    s += variables.yellow

                else:
                    s += variables.green

                s += tex + variables.green
        return s + variables.white

    def depth_first(self):
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def breadth_first(self):
        current_nodes = [self]
        while len(current_nodes) > 0:
            children = []
            for node in current_nodes:
                yield node
                children.extend(node.children)
            current_nodes = children

    def add(self, data):
        return Node(data, parent=self)

    def make_root(self):
        if not self.is_root():
            self.parent.children.remove(self)  # just to be consistent
            self.parent = None
            old_depth = self.depth
            for node in self.breadth_first():
                node.depth -= old_depth

    def find_root(self):
        if self.is_root():
            return self

        else:
            return self.parent.find_root()

    def get_values(self):
        return [child.value for child in self.children]


class Tree:
    """
    Although a Node is a tree by itself, this class provides more iterators and
    quick access to the different depths of
    the tree, and keeps track of the root node
    """

    def __init__(self, root_data):
        self.root = Node(root_data)
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)

    def __len__(self):
        return len(self.nodes)

    def str_tree(self,
                 current_node=None,
                 next_node=None,
                 str_data_fn=lambda node: str(node.data) +
                 ". depth : " + str(node.depth)):

        return self.root.str_node(current_node, next_node, str_data_fn) + \
               " max depth = " + str(self.max_depth)

    def new_root(self, node):
        node.make_root()
        self.root = node
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)

        for n in self.root.breadth_first():
            # iterate through children nodes and add them to the depth list
            self.update(n)

    def update(self, node):
        """
        updates the depth, the nodes list and max_depth
        :param node:
        """
        self.depth[node.depth].append(node)
        self.nodes.append(node)
        if node.depth > self.max_depth:
            self.max_depth = node.depth

    def add(self, parent_node, data):
        """
        *Deprecated*

        Look at the node in the tree.
        If it does not exist, create it.
        If it exists above, do nothing.
        If it exists below, cut it and add it to the parent_node
        """
        for node in self.root.depth_first():
            if node.data == data:
                node.number_visits += 1
                if node.parent == parent_node:  # node exists: do nothing
                    return node

                elif node.depth > parent_node.depth + 1:  # node is below = ?
                    return self.add_tree(parent_node, node)

                else:  # node is above
                    return node

        child = parent_node.add(data)  # node does not exist
        self.update(child)
        return child

    def add_tree(self, parent_node, node):
        """
        add the tree under the parent_node with the right depths
        """
        if not node.is_root():
            node.parent.children.remove(node)  # just to be consistent

        node.parent = parent_node
        old_depth = node.depth
        parent_node.children.append(node)
        for child in node.depth_first():
            child.depth = child.depth - old_depth + (parent_node.depth + 1)

        self.new_root(self.root)  # compute the new depths
        return node

    @staticmethod
    def get_leaves(node):
        """
        new idea : track the list of leaves at all time.
        if root : return all the leaves.
        else: get all the parents for each leaf and compare them to node.
        If node is found, add this leaf to leaves.
        """
        leaves = []
        for child in node.depth_first():
            if child.is_leaf() and (not child.is_root()):
                leaves.append(child)

        return leaves

    @staticmethod
    def get_next_option_index(node, leaf):
        while leaf.parent != node:
            leaf = leaf.parent

        return leaf.parent.children.index(leaf)

    @staticmethod
    def get_probability_leaves(node):

        assert not(node.is_leaf())

        leaves = Tree.get_leaves(node)
        probability_leaves = np.zeros(len(leaves))
        idx = -1
        for leaf in leaves:
            idx += 1
            probability_leaves[idx] = (leaf.depth - node.depth)

        probability_leaves /= np.sum(probability_leaves)

        return probability_leaves, leaves

    @staticmethod
    def get_random_next_option_index(node):
        probability_leaves, leaves = Tree.get_probability_leaves(node)
        selected_leaf = leaves[sample_pmf(probability_leaves)]
        return Tree.get_next_option_index(node, selected_leaf)
