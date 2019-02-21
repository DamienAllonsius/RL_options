import numpy as np
from collections import defaultdict

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
