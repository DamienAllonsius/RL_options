from planning.tree import Node, Tree
import numpy as np
import unittest


class TreeTest(unittest.TestCase):

    def setUp(self):
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
        self.node_8 = Node(data=8)

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

    # ------------- The tests are defined here --------------

    def test_print_tree(self):
        print(self.tree.str_tree())

    def test_new_root(self):
        self.tree.new_root(self.node_3)

        tree = Tree(0)
        tree.root = self.node_3
        tree.nodes = [self.node_3, self.node_6]
        tree.depth[0].append(self.node_3)
        tree.depth[1].append(self.node_6)
        tree.max_depth = 1

        self.assertEqual(self.tree.root, tree.root)
        self.assertEqual(self.tree.nodes, tree.nodes)
        self.assertEqual(self.tree.depth, tree.depth)
        self.assertEqual(self.tree.max_depth, tree.max_depth)

    def test_update(self):
        self.node_8.depth = 3
        self.tree.update(self.node_8)
        self.assertEqual(self.tree.depth[3], [self.node_7, self.node_8])

    def test_add_tree(self):
        self.tree.add_tree(parent_node=self.node_6, node=self.node_8)
        self.assertEqual(self.tree.depth[3], [self.node_7, self.node_8])

    def test_get_leaves(self):
        leaves = self.tree.get_leaves(node=self.tree.root)
        self.assertEqual(leaves, [self.node_7, self.node_5, self.node_2, self.node_6])

    def test_get_next_option_index(self):
        next_node_index_1 = Tree.get_next_option_index(self.tree.root, self.node_4)
        next_node_index_4 = Tree.get_next_option_index(self.node_1, self.node_7)
        next_node_index_3 = Tree.get_next_option_index(self.tree.root, self.node_6)

        self.assertEqual(next_node_index_1, 0)
        self.assertEqual(next_node_index_4, 0)
        self.assertEqual(next_node_index_3, 2)

    def test_get_probability_leaves(self):
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

    def test_get_random_next_option_index(self):
        """

        :return:
        """
        pass
