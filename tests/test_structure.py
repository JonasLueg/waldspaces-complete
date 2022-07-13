""" This is the unittest script for testing the class Structure. """

# external imports
import unittest

# internal imports
from tools.structure_and_split import Split
from tools.structure_and_split import Structure


class TestStructure(unittest.TestCase):
    """ The test class for testing the class Structure. """

    # ----- PARTITIONS TESTS ----- #

    def test_partition_1_1(self):
        # example 1.1
        _n = 1
        p1 = ((0,),)
        p2 = ((0,),)
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_2_1(self):
        # example 2.1
        _n = 2
        p1 = ((1, 0),)
        p2 = ((0, 1),)
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_2_2(self):
        # example 2.2
        _n = 2
        p1 = ((0,), (1,))
        p2 = ((0,), (1,))
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_2_3(self):
        # example 2.3
        _n = 2
        p1 = ((1,), (0,))
        p2 = ((0,), (1,))
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_3_1(self):
        # example 3.1
        _n = 3
        p1 = ((1, 0, 2),)
        p2 = ((0, 1, 2),)
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_3_2(self):
        # example 3.2
        _n = 3
        p1 = ((2, 1), (0,))
        p2 = ((0,), (1, 2))
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_3_3(self):
        # example 3.3
        _n = 3
        p1 = ((0, 2), (1,))
        p2 = ((0, 2), (1,))
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    def test_partition_3_4(self):
        # example 3.4
        _n = 3
        p1 = ((1,), (0,), (2,))
        p2 = ((0,), (1,), (2,))
        self.assert_equality_of_two_partitions(data=[_n, p1, p2])

    # ----- PARTIAL ORDERINGS TESTS ----- #

    def test_partial_order_1_1(self):
        # example 1.1
        _n = 1
        st1 = [((0,),), (tuple(),)]
        st2 = [((0,),), (tuple(),)]
        eq, lt, gt = True, False, False
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_2_1(self):
        # example 2.1
        _n = 2
        st1 = [((0, 1),), (tuple(),)]
        st2 = [((0, 1),), (tuple(),)]
        eq, lt, gt = True, False, False
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_2_2(self):
        # example 2.2
        _n = 2
        st1 = [((0, 1),), ((((0,), (1,)),),)]
        st2 = [((0, 1),), (tuple(),)]
        eq, lt, gt = False, False, True
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_2_3(self):
        # example 2.3
        _n = 2
        st1 = [((0, 1),), ((((0,), (1,)),),)]
        st2 = [((0,), (1,),), (tuple(), tuple())]
        eq, lt, gt = False, False, True
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_4_1(self):
        # example 4.1
        _n = 4
        st1 = [((0, 1, 2, 3),),
               ((((3,), (0, 1, 2)), ((0,), (1, 2, 3)), ((1, 2), (0, 3)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))),)]
        st2 = [((1, 2), (0, 3)),
               ((((1,), (2,)),), (((0,), (3,)),))]
        eq, lt, gt = False, False, True
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_4_2(self):
        # example 4.2
        _n = 4
        st1 = [((0, 1, 2, 3),),
               ((((3,), (0, 1, 2)), ((0,), (1, 2, 3)), ((0, 2), (1, 3)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))),)]
        st2 = [((1, 2), (0, 3)),
               ((((1,), (2,)),), (((0,), (3,)),))]
        eq, lt, gt = False, False, False
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_4_3(self):
        # example 4.3
        _n = 4
        st1 = [((0, 1, 2, 3),),
               ((((3,), (0, 1, 2)), ((0,), (1, 2, 3)), ((0, 2), (1, 3)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))),)]
        st2 = [((0, 1, 2, 3),),
               ((((3,), (0, 1, 2)), ((0,), (1, 2, 3)), ((0, 3), (1, 2)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))),)]
        eq, lt, gt = False, False, False
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    def test_partial_order_4_4(self):
        # example 4.4
        _n = 4
        st1 = [((0, 1, 2, 3),),
               ((((3,), (0, 1, 2)), ((0,), (1, 2, 3)), ((1,), (0, 2, 3)), ((2,), (0, 1, 3))),)]
        st2 = [((1, 2), (0, 3)),
               ((((1,), (2,)),), (((0,), (3,)),))]
        eq, lt, gt = False, False, False
        self.assert_partial_ordering_of_two_structures(data=[_n, st1, st2, eq, lt, gt])

    # ----- HELPING FUNCTIONS FOR TESTING ----- #

    def assert_partial_ordering_of_two_structures(self, data):
        """ data has the form (n, structure1, structure2, true ==, true <, true >). """
        n = data[0]
        f1 = self.make_st(n=n, st=data[1])
        f2 = self.make_st(n=n, st=data[2])
        self.assertEqual(first=(f1 == f2), second=data[3])
        self.assertEqual(first=(f1 < f2), second=data[4])
        self.assertEqual(first=(f1 > f2), second=data[5])

    def assert_equality_of_two_partitions(self, data):
        """ data has the form (n, partition1, partition2). """
        _first = Structure(n=data[0], partition=data[1], split_collection=[tuple() for _ in data[1]]).partition
        _second = data[2]
        self.assertEqual(first=_first, second=_second)

    @staticmethod
    def make_st(n, st):
        partition = st[0]
        split_collection = [[Split(n=n, part1=sp[0], part2=sp[1]) for sp in splits] for splits in st[1]]
        return Structure(n=n, partition=partition, split_collection=split_collection)


