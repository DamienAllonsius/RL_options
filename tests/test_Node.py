import unittest
from planning.tree import Node


class NodeTest(unittest.TestCase):
    def setUp(self):
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
