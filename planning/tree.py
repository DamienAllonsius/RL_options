import numpy as np
from collections import defaultdict
from planning.utils import *

class Node:
    def __init__(self, data, parent=None):
        self.data = data
        self.parent = parent
        if self.parent:
            self.parent.children.append(self)
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0
        self.children = []

    def size(self):
        return np.sum([c.size() for c in self.children]) + 1

    def breadth_first(self):
        current_nodes = [self]
        while len(current_nodes) > 0:
            children = []
            for node in current_nodes:
                yield node
                children.extend(node.children) # extend = append
            current_nodes = children

    def depth_first(self):
        yield self
        for child in self.children:
            for node in child.depth_first():
                yield node

    def is_root(self):
        return self.parent is None

    def is_leaf(self):
        return len(self.children) == 0

    def add(self, data):
        root = self.find_root()
        for node in root.depth_first():
            if node.data == data:
                return node
            
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

    def str_node(self, str_data_fn=lambda data: str(data)):
        tab = '   '
        # '\033[91m' if for printing red text in the console !
        s =  '\033[92m' + str_data_fn(self.data) + '\n'
        for node in self.depth_first():
            
            d = node.depth - self.depth
            if d > 0:
                s += "".join([tab] * d + ["|", str_data_fn(node.data), '\n'])
        return s + '\033[0m'

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

    def str_tree(self, str_data_fn=lambda data: str(data)):
        return (self.root.str_node(str_data_fn))

    def iter_depth_first(self, include_root=False, include_leaves=True):
        iterator = self.root.depth_first()
        try:
            root = next(iterator)
            if include_root:
                yield root
            while True:
                node = next(iterator)
                if include_leaves or not node.is_leaf():
                    yield node
        except StopIteration:
            pass

    def iter_breadth_first(self, include_root=False, include_leaves=True):
        if include_root:
            yield self.root
        for d in range(1, self.max_depth + 1):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node

    def iter_breadth_first_reverse(self, include_root=False, include_leaves=True):
        for d in range(self.max_depth, 0, -1):
            for node in self.depth[d]:
                if include_leaves or not node.is_leaf():
                    yield node
        if include_root:
            yield self.root

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

    def add(self, parent_node, data):
        """
        if the node exits then return it. Else: add it in the regular way
        """
        for node in self.root.depth_first():
            if node.data == data:
                return node

        else:
            child = parent_node.add(data)
            self._add(child)
            return child

    def extract_trajectory(self, node):
        trajectory = [node.data]
        while not node.is_root():
            node = node.parent
            trajectory.append(node.data)
        return list(reversed(trajectory))

    def get_leaves(self):
        leaves = []
        for node in self.root.depth_first():
            if node.is_leaf() and (node.depth != 0):
                leaves.append(node)

        return leaves

    def get_total_depth(self, leaves):
        total_depth = 0
        for node in leaves:
            total_depth += node.depth

        return total_depth

    def get_good_trajectory(self):
        probability_leaves = []
        leaves = self.get_leaves()
        total_depth = self.get_total_depth(leaves)
        for leaf in leaves:
            probability_leaves.append(leaf.depth / total_depth)

        selected_leaf = leaves[sample_pmf(probability_leaves)]
        return self.extract_trajectory(selected_leaf)
    
class TreeActor:
    """
    Interacts with an environment while adding nodes to a tree.
    """

    def __init__(self, env, observe_fn=None):
        self.env = env
        self.tree = None
        self.observe_fn = observe_fn if observe_fn is not None else lambda x: x

    def generate_successor(self, node, action):
        if self.last_node is not node:
            self.env.unwrapped.restore_state(node.data["s"])

        # Perform step
        next_obs, r, end_of_episode, info = self.env.step(action)
        node_data = {"a": action, "r": r, "done": end_of_episode, "obs": next_obs}
        node_data.update(info) # add extra info e.g. atari lives
        child = self.tree.add(node, node_data)
        self._observe(child)
        return child

    def step(self, a, cache_subtree):
        next_node = self._get_next_node(self.tree, a)
        root_data = self.tree.root.data

        # "take a step" (actually remove other branches and make selected child root)
        self.tree.new_root(next_node, keep_subtree=cache_subtree)
        if self.last_node is not self.tree.root:
            self.last_node = None  # just in case, we'll restore before expanding

        return root_data, next_node.data

    def reset(self):
        obs = self.env.reset()
        self.tree = Tree(self.env.action_space.n, {"obs": obs, "done": False})
        self._observe(self.tree.root)
        return self.tree

    def _observe(self, node):
        node.data["s"] = self.env.unwrapped.clone_state()
        self.observe_fn(self.env, node)
        self.last_node = node

    def _get_next_node(self, tree, a):
        assert not tree.root.is_leaf()

        next_node = None
        for child in tree.root.children:
            if a == child.data["a"]:
                next_node = child
        assert next_node is not None, "Selected action not in tree. Something wrong with the lookahead policy?"

        return next_node
