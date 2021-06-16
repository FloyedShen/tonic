import unittest
from parameterized import parameterized
import numpy as np
import torch
import tonic.functional as F
import utils
import math


class TestFunctionalTorch(unittest.TestCase):
    def findXytpPermutation(self, ordering):
        x_index = ordering.find("x")
        y_index = ordering.find("y")
        t_index = ordering.find("t")
        p_index = ordering.find("p")
        return x_index, y_index, t_index, p_index

    @parameterized.expand([("xytp", True), ("typx", False)])
    def testToSparseTensor(self, ordering, merge_polarities):
        (
            events,
            images,
            sensor_size,
            is_multi_image,
        ) = utils.create_random_input_with_ordering(ordering)
        tensor = F.to_sparse_tensor_pytorch(
            events,
            sensor_size=sensor_size,
            ordering=ordering,
            merge_polarities=merge_polarities
        )
        x_index, y_index, t_index, p_index = self.findXytpPermutation(ordering)
        
        self.assertTrue(isinstance(tensor, torch.Tensor))
        self.assertTrue(tensor.is_sparse)
        self.assertEqual(len(tensor.shape), 4, "Tensor should have 4 dimensions, (TCXY)")
        self.assertEqual(events.shape[0], tensor.coalesce().values().sum(), "Sparse tensor values should contain as many 1s as there are events.")
        self.assertEqual(tensor.size()[2:], sensor_size, "Sparse tensor should have sensor size dimensions.")
        if merge_polarities:
            self.assertEqual(tensor.shape[1], 1)
        else:
            self.assertEqual(tensor.shape[1], 2)
