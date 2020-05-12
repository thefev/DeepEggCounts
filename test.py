import unittest
import data
import numpy as np


class TestData(unittest.TestCase):
    def test_coor_rot90_1(self):
        coor = np.array([[1, 5], [100, 300]])
        img_shape = (480, 360, 3)
        a = data.coor_rot90(img_shape, coor)
        e = np.array([[475, 1], [180, 100]])
        self.assertTrue((a == e).all())

    def test_coor_rot90_2(self):
        coor = np.array([[295, 63], [349, 156], [123, 321]])
        img_shape = (360, 480, 3)
        a = data.coor_rot90(img_shape, coor)
        e = np.array([[297, 295], [204, 349], [39, 123]])
        self.assertTrue((a == e).all())


if __name__ == '__main__':
    unittest.main()
