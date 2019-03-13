import numpy as np
from collections import defaultdict
from planning.utils import *
#from utils import *
import time

class Node:
    def __init__(self, data, parent=None):
        self.values = []
        self.data = data #a.k.a state
        self.actions = []
        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
            
        else:
            self.depth = 0
            
        self.children = []

    def __str__(self):
        tab = "    "
        s = str(self.data) + " at depth " + str(self.depth) + "\n"
        if self.parent != None:
            s += "parent: " + str(self.parent.data) + "\n"
            
        s += "node and children:" + "\n"
        s += self.str_node(self)
        for k in range(len(self.actions)):
            s += "action: " +str(self.actions[k]) + " value: " + str(self.values[k]) + "\n"
        return s
        
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

    def str_node(self, current_node = None, next_node = None, str_data_fn=lambda node: str(node.data)):
        red = '\033[91m'
        green = '\033[92m'
        yellow = '\033[93m'
        white = '\033[0m'
        tab = '   '
        # '\033[91m' if for printing red text in the console !
        if self.data == current_node:
            s = red

        else:
            s = green

        s += str_data_fn(self) + '\n'
        for node in self.depth_first():
            d = node.depth - self.depth
            if d > 0:
                tex = "".join([tab] * d + ["|", str_data_fn(node), '\n'])
                if node.data == current_node:
                    s += red
                    
                elif node.data == next_node:
                    s += yellow

                else:
                    s += green
                    
                s += tex + green
        return s + white

# a = Node(1)
# b = a.add(2)
# c = a.add(3)
# d = a.add(4)
# e = c.add(5)
# c.add(6)

# for node in a.depth_first():
#     print(node.data)

class Tree:
    """
    Although a Node is a tree by itself, this class provides more iterators and quick access to the different depths of
    the tree, and keeps track of the root node
    """

    def __init__(self, root_data):
        self.new_root(Node(root_data))

    def __len__(self):
        return len(self.nodes)

    def str_tree(self, current_node = None, next_node = None, str_data_fn=lambda node: str(node.data) + ". depth : " + str(node.depth)):
        return (self.root.str_node(current_node, next_node, str_data_fn) + " max depth = " + str(self.max_depth))

    def new_root(self, node, keep_subtree=True):
        node.make_root()
        self.root = node
        self.max_depth = 0
        self.nodes = list()
        self.depth = defaultdict(list)
        if not keep_subtree:
            node.children = list()  # remove children
        for n in self.root.breadth_first():
            self._add(n)  # iterate through children nodes and add them to the depth list

    def _add(self, node):
        self.depth[node.depth].append(node)
        self.nodes.append(node)
        if node.depth > self.max_depth: self.max_depth = node.depth

    def add_tree(self, parent_node, node):
        """
        add the tree under the parent_node with the right depths
        """
        node.parent.children.remove(node)  # just to be consistent
        node.parent = parent_node
        old_depth = node.depth
        parent_node.children.append(node)
        for child in node.depth_first():
            child.depth = child.depth - old_depth + (parent_node.depth + 1)

        self.new_root(self.root, keep_subtree = True) # compute the new depths
        return node

    def add(self, parent_node, data):
        """
        Look at the node in the tree. 
        If it does not exist, create it.
        If it exists above, do nothing.
        If it exists below, cut it and add it to the parent_node
        """
        for node in self.root.depth_first():
            if node.data == data:
                if node.parent == parent_node: # node exists: do nothing
                    return node
                
                elif node.depth > parent_node.depth + 1: # node is below = ?
                    return self.add_tree(parent_node, node)

                else: # node is above
                    return node

        child = parent_node.add(data) # node does not exist
        self._add(child)
        return child

    def get_leaves(self, node):
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

    def get_total_depth(self, node, leaves):
        total_depth = 0
        for leaf in leaves:
            total_depth += leaf.depth - node.depth

        return total_depth

    def get_next_data(self, node, leaf):
        last_data = leaf.data
        while leaf != node:
            last_data = leaf.data
            leaf = leaf.parent
            
        return last_data

    def get_random_next_data(self, node):
        if node.is_leaf():
            return None
        
        probability_leaves = []
        leaves = self.get_leaves(node)
        total_depth = self.get_total_depth(node, leaves)
        for leaf in leaves:
            probability_leaves.append((leaf.depth - node.depth) / total_depth)

        selected_leaf = leaves[sample_pmf(probability_leaves)]
        return self.get_next_data(node, selected_leaf)

# tree = Tree(0)
# tree.add(tree.root, 12)
# b = tree.add(tree.root, 4)
# c = tree.add(tree.root, 2)
# tree.add(b, 1)
# tot = tree.add(b, 8)
# ac = tree.add(tot, 10)
# z = tree.add(ac, 5)
# tree.add(z, 3)
# bubu = tree.add(ac, 124)
# tree.add(c, 11)
# tree.add(tree.root, 11)
# tree.add(tree.root, 1)
# tree.add(b, 124)
# tree.add(tree.root, 8)
# tree.add(b,10)
# print(tree.str_tree())

# node = tree.root
# while not node.is_leaf():
#     print(node.data)
#     node = tree.get_random_next_data(node)
