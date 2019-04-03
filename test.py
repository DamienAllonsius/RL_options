import unittest
from planning.tree import Node
from planning.tree import Tree
import numpy as np


class NodeTest(unittest.TestCase):
    def make_data(self):
        """
        We define here some nodes to test their functions
        """
        self.node_0 = Node(data=0)
        self.node_1 = Node(data=1)
        self.node_2 = Node(data=2)
        self.node_3 = Node(data=3)
        self.node_4 = Node(data=4)
        self.node_5 = Node(data=5)
        self.node_6 = Node(data=6)
        self.node_7 = Node(data=7)

        self.set_parents_children()
        self.set_values()

    def set_parents_children(self):
        """
        Defines a Tree with the nodes
        :return:
        """
        self.node_0.children = [self.node_1, self.node_2, self.node_3]
        self.node_1.children = [self.node_4, self.node_5]
        self.node_3.children = [self.node_6]
        self.node_4.children = [self.node_7]

    def set_values(self):
        self.node_0.value = 0
        self.node_1.value = 1
        self.node_2.value = 10
        self.node_3.value = 11
        self.node_4.value = 100
        self.node_5.value = 101
        self.node_6.value = 111
        self.node_7.value = 1000

    def test_get_values(self):
        self.make_data()
        values_0 = self.node_0.get_values()
        values_1 = self.node_1.get_values()
        values_2 = self.node_2.get_values()
        values_3 = self.node_3.get_values()
        values_7 = self.node_7.get_values()

        self.assertEqual(values_0, [1, 10, 11])
        self.assertEqual(values_1, [100, 101])
        self.assertEqual(values_2, [])
        self.assertEqual(values_3, [111])
        self.assertEqual(values_7, [])


class TreeTest(unittest.TestCase):

    def make_data(self):
        """
        We define here a Tree to test its functions
        """
        self.tree = Tree(root_data=0)
        self.node_1 = Node(data=1)
        self.node_2 = Node(data=2)
        self.node_3 = Node(data=3)
        self.node_4 = Node(data=4)
        self.node_5 = Node(data=5)
        self.node_6 = Node(data=6)
        self.node_7 = Node(data=7)

        self.set_parents_children()
        self.set_values()

    def set_values(self):
        self.tree.root.value = 0
        self.node_1.value = 1
        self.node_2.value = 10
        self.node_3.value = 11
        self.node_4.value = 100
        self.node_5.value = 101
        self.node_6.value = 111
        self.node_7.value = 1000

    def set_parents_children(self):
        """
        Defines a Tree with the nodes
        :return:
        """
        self.tree.add_tree(self.tree.root, self.node_1)
        self.tree.add_tree(self.tree.root, self.node_2)
        self.tree.add_tree(self.tree.root, self.node_3)

        self.tree.add_tree(self.node_1, self.node_4)
        self.tree.add_tree(self.node_1, self.node_5)

        self.tree.add_tree(self.node_3, self.node_6)

        self.tree.add_tree(self.node_4, self.node_7)

    def test_get_probability_leaves(self):
        self.make_data()
        leaves_0, _ = Tree.get_probability_leaves(self.tree.root)
        leaves_1, _ = Tree.get_probability_leaves(self.node_1)
        leaves_3, _ = Tree.get_probability_leaves(self.node_3)
        leaves_4, _ = Tree.get_probability_leaves(self.node_4)

        with self.assertRaises(Exception):
            leaves_2, _ = self.tree.get_probability_leaves(self.node_2)
        with self.assertRaises(Exception):
            leaves_5, _ = self.tree.get_probability_leaves(self.node_5)
        with self.assertRaises(Exception):
            leaves_7, _ = self.tree.get_probability_leaves(self.node_7)
        with self.assertRaises(Exception):
            leaves_6, _ = self.tree.get_probability_leaves(self.node_7)

        np.testing.assert_array_equal(leaves_0, np.array([3 / 8, 2 / 8, 1 / 8, 2 / 8]))
        np.testing.assert_array_equal(leaves_1, np.array([2 / 3, 1 / 3]))
        np.testing.assert_array_equal(leaves_3, np.array([1]))
        np.testing.assert_array_equal(leaves_4, np.array([1]))

    def test_print_tree(self):
        self.make_data()
        print(self.tree.str_tree())

    def test_get_next_data(self):
        """
        TODO
        """

    def test_get_random_next_data(self):
        """
        TODO
        """


class QTest(unittest.TestCase):
    def test_get_number_options(self):
        """
        TODO
        """


class OptionTest(unittest.TestCase):
    pass


class AgentTest(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
