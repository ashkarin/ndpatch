import unittest
import numpy as np
import ndpatch


class TestFindSegments(unittest.TestCase):
    def test_find_segments_patch_inside(self):
        """
        Tests correctness of grid and patch segments.
        The patch is inside the grid cell corresponding to the actual data array.
        """
        # () - does not include boundary, [] - includes boundary
        # find grid:        -3   -2  -1   0  1  2  3  4  5  6  7  8  9  10  11  ...
        # coarse cell:      ~~   ~~  ~~  [0  1  2  3  4  5  6  7  8  9] ~~  ~~  ...
        # grid segments:    -3   -2  -1   0  1 [2  3  4  5  6  7) 8  9  10  11  ...
        # patch segments:                      [0  1  2  3  4  5)
        coarse_cell_size = 10
        patch_size = 5
        index = 2
        gird_segments, patch_segments = ndpatch.find_segments(coarse_cell_size, patch_size, index)
        
        self.assertEqual(len(gird_segments), 1)
        self.assertEqual(len(patch_segments), 1)

        self.assertSequenceEqual(gird_segments[0], (2, 7))
        self.assertSequenceEqual(patch_segments[0], (0, 5))
    
    def test_find_segments_patch_partially_outside(self):
        """
        Tests correctness of grid and patch segments.
        The patch is partially outside the grid cell corresponding to the actual data array.
        """
        # () - does not include boundary, [] - includes boundary
        # find grid:        -3   -2  -1   0  1  2  3  4  5  6  7  8  9  10  11  ...
        # coarse cell:      ~~   ~~  ~~  [0  1  2  3  4  5  6  7  8  9] ~~  ~~  ...
        # grid segments:    -3  [-2  -1  [0) 1  2  3) 4  5  6  7  8  9  10  11  ...
        # patch segments:       [ 0   1  [2) 3  4  5)
        coarse_cell_size = 10
        patch_size = 5
        index = -2
        gird_segments, patch_segments = ndpatch.find_segments(coarse_cell_size, patch_size, index)
        
        self.assertEqual(len(gird_segments), 2)
        self.assertEqual(len(patch_segments), 2)

        self.assertSequenceEqual(gird_segments[0], (-2, 0))
        self.assertSequenceEqual(gird_segments[1], (0, 3))
        self.assertSequenceEqual(patch_segments[0], (0, 2))
        self.assertSequenceEqual(patch_segments[1], (2, 5))
    
    def test_find_segments_patch_outside(self):
        """
        Tests correctness of grid and patch segments.
        The patch is outside the grid cell corresponding to the actual data array.
        """
        # () - does not include boundary, [] - includes boundary
        # find grid:        -20  -19  -18  -17  -16  -15  -14  -13  -12  ...
        # coarse cell:      ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ~~~  ...
        # grid segments:   [-20  -19  -18  -17  -16  -15)  -14  -13  -12 ...
        # patch segments:  [  0    1    2    3    4    5)
        coarse_cell_size = 10
        patch_size = 5
        index = -20
        gird_segments, patch_segments = ndpatch.find_segments(coarse_cell_size, patch_size, index)
        
        self.assertEqual(len(gird_segments), 1)
        self.assertEqual(len(patch_segments), 1)

        self.assertSequenceEqual(gird_segments[0], (-20, -15))
        self.assertSequenceEqual(patch_segments[0], (0, 5))

    def test_find_segments_patch_includes(self):
        """
        Tests correctness of grid and patch segments.
        The patch includes the grid cell corresponding to the actual data array.
        """
        # () - does not include boundary, [] - includes boundary
        # find grid:        -3   -2  -1   0  1  2  3  4  5  6  7  8  9   10  11  12  ...
        # coarse cell:      ~~   ~~  ~~  [0  1  2  3  4] 5  6  7  8  9   ~~  ~~  ~~  ...
        # grid segments:    -3  [-2  -1  [0) 1  2  3  4 [5) 6  7) 8  9  10  11  12  ...
        # patch segments:       [ 0   1  [2) 3  4  5  6 [7) 8  9)
        coarse_cell_size = 5
        patch_size = 9
        index = -2
        gird_segments, patch_segments = ndpatch.find_segments(coarse_cell_size, patch_size, index)
        self.assertEqual(len(gird_segments), 3)
        self.assertEqual(len(patch_segments), 3)

        self.assertSequenceEqual(gird_segments[0], (-2, 0))
        self.assertSequenceEqual(gird_segments[1], (0, 5))
        self.assertSequenceEqual(gird_segments[2], (5, 7))
        self.assertSequenceEqual(patch_segments[0], (0, 2))
        self.assertSequenceEqual(patch_segments[1], (2, 7))
        self.assertSequenceEqual(patch_segments[2], (7, 9))
    
    def test_find_segments_patch_equals(self):
        """
        Tests correctness of grid and patch segments.
        The patch equals to the grid cell corresponding to the actual data array.
        """
        # () - does not include boundary, [] - includes boundary
        # find grid:        -3   -2  -1   0  1  2  3  4  5  6  7  ...
        # coarse cell:      ~~   ~~  ~~  [0  1  2  3  4] 5  6  7  ...
        # grid segments:    -3   -2  -1  [0  1  2  3  4  5) 6  7  ...
        # patch segments:                [0  1  2  3  4  5)
        coarse_cell_size = 5
        patch_size = 5
        index = 0
        gird_segments, patch_segments = ndpatch.find_segments(coarse_cell_size, patch_size, index)
        
        self.assertEqual(len(gird_segments), 1)
        self.assertEqual(len(patch_segments), 1)

        self.assertSequenceEqual(gird_segments[0], (0, 5))
        self.assertSequenceEqual(patch_segments[0], (0, 5))


class TestGet2DNDPatch(unittest.TestCase):
    def test_get_patch_inside(self):
        """
        Tests correctness of 2D patch extraction.
        The patch is inside the grid cell corresponding to the actual data array.
        """
        array = np.random.random((10,10))
        
        index = (2, 2)
        patch_shape = (5, 5)
        patch = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (patch == array[2:7, 2:7])
        self.assertTrue(np.all(equal))

    def test_get_patch_partially_outside(self):
        """
        Tests correctness of 2D patch extraction.
        The patch is partially outside the grid cell corresponding to the actual data array.
        """
        # Quadrants (M - mirrored, R - real):
        #  M M
        #  M R
        #  Center of the grid at the top left corner of R quadrant
        #
        #   -5  -4  -3  -2  -1 |  0   1   2   3   4    Coordinates
        #   _________________________________________
        # | 24  23  22  21  20 | 20  21  22  23  24 | -5
        # | 19  18  17  16  15 | 15  16  17  18  19 | -4
        # | 14  13  12  11  10 | 10  11  12  13  14 | -3
        #           --------patch------
        # |  9   8 | 7   6   5 |  5   6|  7   8   9 | -2
        # |  4   3 | 2   1   0 |  0   1|  2   3   4 | -1 <--- Mirrored (virtual)
        #  __________________________________________
        # |  4   3 | 2   1   0 |  0   1|  2   3   4 |  0 <--- Real array
        # |  9   8 | 7   6   5 |  5   6|  7   8   9 |  1
        # | 14  13 |12  11  10 | 10  11| 12  13  14 |  2
        # |         --------patch------
        # | 19  18  17  16  15 | 15  16  17  18  19 |  3
        # | 24  23  22  21  29 | 20  21  22  23  24 |  4 
        #  _________________________________________
        array = np.arange(0, 25).reshape((5,5))

        patch_shape = (5, 5)
        index = (-2, -3)
        expected = np.array([
            [ 7,  6,  5,  5,  6],
            [ 2,  1,  0,  0,  1],
            [ 2,  1,  0,  0,  1],
            [ 7,  6,  5,  5,  6],
            [12, 11, 10, 10, 11]
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))

        patch_shape = (5, 7)
        index = (3, 2)
        expected = np.array([
            [ 17, 18, 19, 19, 18, 17, 16],
            [ 22, 23, 24, 24, 23, 22, 21],
            [ 22, 23, 24, 24, 23, 22, 21],
            [ 17, 18, 19, 19, 18, 17, 16],
            [ 12, 13, 14, 14, 13, 12, 11],
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))

        patch_shape = (5, 6)
        index = (-7, -8)
        expected = np.array([
            [17, 18, 19, 19, 18, 17],
            [22, 23, 24, 24, 23, 22],
            [22, 23, 24, 24, 23, 22],
            [17, 18, 19, 19, 18, 17],
            [12, 13, 14, 14, 13, 12]
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))

    def test_get_patch_outside(self):
        """
        Tests correctness of 2D patch extraction.
        The patch is partially outside the grid cell corresponding to the actual data array.
        """
        # Quadrants (M - mirrored, R - real):
        #  M M
        #  M R
        #  Center of the grid at the top left corner of R quadrant
        #
        #   -5  -4  -3  -2  -1 |  0   1   2   3   4    Coordinates
        #  __________________________________________
        # | 24  23  22  21  20 | 20  21  22  23  24 | -5
        #       ---patch---
        # | 19 |18  17  16| 15 | 15  16  17  18  19 | -4
        # | 14 |13  12  11| 10 | 10  11  12  13  14 | -3
        # |  9 | 8   7   6|  5 |  5   6   7   8   9 | -2
        #       ---patch---
        # |  4   3   2   1   0 |  0   1   2   3   4 | -1 <--- Mirrored (virtual)
        #  __________________________________________
        # |  4   3   2   1   0 |  0   1   2   3   4 |  0<--- Real array
        # |  9   8   7   6   5 |  5   6   7   8   9 |  1
        # | 14  13  12  11  10 | 10  11  12  13  14 |  2
        # | 19  18  17  16  15 | 15  16  17  18  19 |  3
        # | 24  23  22  21  29 | 20  21  22  23  24 |  4
        #  _________________________________________
        array = np.arange(0, 25).reshape((5,5))
        
        patch_shape = (3, 3)
        index = (-4, -4)
        expected = np.array([
            [ 18, 17, 16],
            [ 13, 12, 11],
            [ 8, 7, 6]
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))

        patch_shape = (3, 4)
        index = (-10, -9)
        expected = np.array([
            [ 1, 2, 3, 4],
            [ 6, 7, 8, 9],
            [ 11, 12, 13, 14]
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))

        patch_shape = (2, 2)
        index = (10, 5)
        expected = np.array([
            [ 4, 3],
            [ 9, 8]
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))

    def test_get_patch_includes(self):
        """
        Tests correctness of 2D patch extraction.
        The patch includes the grid cell corresponding to the actual data array.
        """
        array = np.arange(0, 9).reshape((3,3))

        patch_shape = (6, 7)
        index = (-1, -2)
        expected = np.array([
            [1, 0, 0, 1, 2, 2, 1],
            [1, 0, 0, 1, 2, 2, 1],
            [4, 3, 3, 4, 5, 5, 4],
            [7, 6, 6, 7, 8, 8, 7],
            [7, 6, 6, 7, 8, 8, 7],
            [4, 3, 3, 4, 5, 5, 4]
        ])
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (expected == extracted)
        self.assertTrue(np.all(equal))
    
    def test_get_patch_equals(self):
        """
        Tests correctness of 2D patch extraction.
        The patch equals to the grid cell corresponding to the actual data array.
        """
        array = np.arange(0, 9).reshape((3,3))
        patch_shape = (3, 3)
        index = (0, 0)
        extracted = ndpatch.get_ndpatch(array, patch_shape, index)
        equal = (array == extracted)
        self.assertTrue(np.all(equal))


class TestGetPatchesIndices(unittest.TestCase):
    def test_get_patches_indices_wrong_overlap(self):
        array_shape = (20, 20)
        patch_shape = (3, 3)
        
        # Too large
        overlap = 6
        self.assertRaises(ValueError, ndpatch.get_patches_indices, array_shape, patch_shape, overlap)
        
        # Too large
        overlap = (6, 2)
        self.assertRaises(ValueError, ndpatch.get_patches_indices, array_shape, patch_shape, overlap)
        
        # Negative
        overlap = (2, -2)
        self.assertRaises(ValueError, ndpatch.get_patches_indices, array_shape, patch_shape, overlap)


class TestReconstructFromPatches(unittest.TestCase):
    def test_reconstruct_from_patches_1D(self):
        """
        Tests correctness of the array reconstruction from patches (2D).
        """
        array = np.arange(0, 5)
        patch_shape = 3
        overlap = 0
        indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
        patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
        reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
        equal = (array == reconstructed)
        self.assertTrue(np.all(equal))
    
    def test_reconstruct_from_patches_2D(self):
        """
        Tests correctness of the array reconstruction from patches (2D).
        """
        array = np.arange(0, 25).reshape((5,5))

        patch_shape = (3, 3)
        overlap = 0
        indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
        patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
        reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
        equal = (array == reconstructed)
        self.assertTrue(np.all(equal))
    
    def test_reconstruct_from_patches_3D(self):
        """
        Tests correctness of the array reconstruction from patches (2D).
        """
        array = np.arange(0, 125).reshape((5,5,5))

        patch_shape = (3, 3, 3)
        overlap = 0
        indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
        patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
        reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
        equal = (array == reconstructed)
        self.assertTrue(np.all(equal))

    def test_reconstruct_from_patches_2D_overlap(self):
        """
        Tests correctness of the array reconstruction from patches (2D).
        """
        array = np.arange(0, 25).reshape((5,5))

        patch_shape = (3, 3)
        overlap = 2
        indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
        patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
        reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
        equal = (array == reconstructed)
        self.assertTrue(np.all(equal))
    
    def test_reconstruct_from_patches_3D_overlap(self):
        """
        Tests correctness of the array reconstruction from patches (2D).
        """
        array = np.arange(0, 125).reshape((5,5,5))

        patch_shape = (3, 3, 3)
        overlap = 2
        indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
        patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
        reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
        equal = (array == reconstructed)
        self.assertTrue(np.all(equal))
    
    def test_reconstruct_from_patches_4D_overlap(self):
        """
        Tests correctness of the array reconstruction from patches (2D).
        """
        array = np.arange(0, 500).reshape((5,5,5,4))

        patch_shape = (3, 3, 3, 4)
        overlap = 2
        indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
        patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
        reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
        equal = (array == reconstructed)
        self.assertTrue(np.all(equal))
