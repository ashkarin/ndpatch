import itertools
import collections
import numpy as np
from math import ceil, floor

try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence


def _is_sequence_of_integers(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return isinstance(value, Sequence) and np.all(isinstance(v, int) for v in value)


def get_patches_indices(array_shape, patch_shape, overlap=None, start=None):
    """Splits the array to patches and returns their corner indices.
    
    Arguments:
        array_shape (sequence) -- shape of the array
        patch_shape (sequence / int) -- shape of patches / size of a patch in each dimension
    
    Keyword Arguments:
        overlap (tuple / int) -- overlap for each dimension / size of overlap in each dimension (default: None)
        start (tuple) -- starting position (default: None)
    """
    if not _is_sequence_of_integers(array_shape):
        raise TypeError("'array_shape' should be a sequence of integer values")

    # Patch shape
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape,) * len(array_shape)

    if not _is_sequence_of_integers(patch_shape):
        raise TypeError("'patch_shape' should be a sequence of integer values or an integer value. | type(patch_shape)={}".format(type(patch_shape)))
    
    if len(patch_shape) != len(array_shape):
        raise RuntimeError("The lengths of 'patch_shape' and 'array_shape' are not the same. | patch_shape={}, array_shape={}".format(patch_shape, array_shape))
    
    # Overlap
    if overlap is None:
        overlap = np.zeros_like(array_shape)
    
    if isinstance(overlap, int):
        overlap = (overlap,) * len(array_shape)
    
    if not _is_sequence_of_integers(overlap):
        raise TypeError("'overlap' should be a sequence of unsigned integer values or an unsigned integer value. | type(overlap)={}".format(type(overlap)))

    if any(v >= s or v < 0 for v, s in zip(overlap, patch_shape)):
        raise ValueError("'overlap' values should be positive integers smaller than corresponding values of 'patch_shape'. | overlap={}, patch_shape={}".format(overlap, patch_shape))

    if len(overlap) != len(array_shape):
        raise RuntimeError("The lengths of 'overlap' and 'array_shape' are not the same. | overlap={}, array_shape={}".format(overlap, array_shape))

    # Start
    if start is None:
        array_shape = np.asarray(array_shape)
        patch_shape = np.asarray(patch_shape)
        overlap = np.asarray(overlap)
        n_patches = np.ceil(array_shape.astype(np.float) / (patch_shape - overlap)).astype(np.int32)
        overflow = (patch_shape - overlap) * n_patches - array_shape + overlap
        start = -np.ceil(overflow.astype(np.float) / 2).astype(np.int32)
        start = tuple(start)

    if isinstance(start, int):
        start = (0,) * len(array_shape)

    if not _is_sequence_of_integers(start):
        raise TypeError("'start' should be a sequence of integer values or integer value. | type(start) = {}".format(type(start)))
    
    if len(start) != len(array_shape):
        raise RuntimeError("The lengths of 'start' and 'array_shape' are not the same. | start={}, array_shape={}".format(start, array_shape))

    # Get indices
    stop = start + array_shape
    step = patch_shape - overlap
    slices = [slice(_start, _stop, _step) for _start, _stop, _step in zip(start, stop, step)]
    return np.array(np.mgrid[slices].reshape(len(slices), -1).T, dtype=np.int).tolist()

def reconstruct_from_patches(patches, indices, array_shape, default_value=0, average=True):
    """Reconstruct an array from the patches.
    
    Arguments:
        patches (list) -- patches to reconstruct the array
        indices (list) -- indices of patches on the find grid (can be negative)
        array_shape (tuple) -- shape of the array (defines a coarse cell of the grid)
    
    Keyword Arguments:
        default_value (float) -- value to fill the array (default: 0)
        average (bool) -- average the values of overlapping patches (default: True)

    Returns:
        ndarray -- reconstructed array
    """
    if not _is_sequence_of_integers(array_shape):
        raise TypeError("'array_shape' should be a sequence of integer values")

    array = np.ones(array_shape) * default_value
    counter = np.zeros(array_shape, dtype=np.uint32)

    patch_start = np.zeros_like(array_shape)
    patch_stop = np.zeros_like(array_shape)

    array_start = np.zeros_like(array_shape)
    array_stop = np.zeros_like(array_shape)

    array_shape = np.array(array_shape)
    for patch, index in zip(patches, indices):
        start = np.array(index)
        stop = np.array([i + s for i, s in zip(start, patch.shape)])

        # Skip the patch located outside the array
        if np.any(stop < 0) or np.any(start > array_shape):
            continue
        
        # Crop the patch if it is partly outside the array
        patch_start.fill(0)
        patch_stop[:] = patch.shape
        patch_start[start < 0] = np.abs(start[start < 0])
        patch_stop[stop > array_shape] -= (stop - array_shape)[stop > array_shape]

        array_start.fill(0)
        array_stop[:] = array_shape
        array_start[start > 0] = start[start > 0]
        array_stop[stop < array_shape] = stop[stop < array_shape]

        # Make ROIs
        patch_roi = tuple(slice(b, e) for b, e in zip(patch_start, patch_stop))
        array_roi = tuple(slice(b, e) for b, e in zip(array_start, array_stop))

        # Update array
        if average:
            array[array_roi] = array[array_roi] * (counter[array_roi] > 0) + patch[patch_roi]
            counter[array_roi] += 1
        else:
            array[array_roi] = patch[patch_roi]
    
    # Average values if required
    if average:
        array = array / counter

    return array


def get_random_nd_index(shape):
    """Get random n-dimensional index in n-dimensional space.
    
    Arguments:
        shape (tuple) -- shape of the space
    
    Returns:
        tuple -- index
    """
    if not _is_sequence_of_integers(shape):
        raise TypeError("'shape' should be a sequence of integer values")

    return tuple([np.random.choice(shape[index] + 1) for index in range(len(shape))])


def get_random_patch_index(array_shape, patch_shape):
    """Get a random corner index to extract a region (patch) from the data array.

    If this is used during neural network training, the middle values will be seen by
    the model way more often than the edge values (which is probably a bad thing).
    
    Arguments:
        array_shape (tuple) -- shape of the array
        patch_shape (tuple) -- shape of the patch
    
    Returns:
        (tuple) -- corner index
    """
    return get_random_nd_index(np.subtract(array_shape, patch_shape))


def find_segments(array_size, patch_size, index):
    """ Find grid and patch segmentes.
    
    There is an infinite grid, which coarse cells have array shape. 
    The index represents an arbitrary position on the fine grid. 
    The region can be larger or smaller than a coarse cell, but it has as many segments
    as the number of covered coarse cells.
    
    Arguments:
        array_size (uint) -- array size in the given dimension
        patch_size (uint) -- patch size in the given dimension 
        index (int) -- index on the fine grid (can be negative)

    Returns:
        (list, list) -- grid and patch segments
    """
    start, stop = index, index + patch_size
    grid_segments = []
    patch_segments = []

    # Set a position and compute the number of segments
    # falling in the region (if region larger than data)
    grid_pos = start
    patch_pos = 0
    nsegments = int(ceil(float(patch_size) / array_size))

    for _ in range(0, nsegments + 1):
            # Compute next closest cell boundary
            boundary = (int(floor(float(grid_pos) / array_size)) + 1) * array_size
            
            # Reset the boundary if the region ends before it.
            boundary = stop if stop <= boundary else boundary

            # Add grid segment
            grid_segments.append((grid_pos, boundary))

            # Compute segment size
            segment_size = abs(boundary - grid_pos)

            # Add region segment
            patch_segments.append((patch_pos, patch_pos + segment_size))

            if boundary == stop:
                break
            
            # Update the grid and region position
            grid_pos = boundary
            patch_pos = patch_pos + segment_size
    
    return grid_segments, patch_segments


def _segments2slices(array_size, grid_segments, patch_segments):
    """Convert segments to slices.
    
    Arguments:
        array_size (uint) -- array size in the given dimension
        patch_segments (list) -- grid segments in the given dimension
        region_segments (list) -- region segments in the given dimension
    
    Returns:
        (list, list) -- array and patch slices
    """
    patch_slices = [slice(start, stop) for start, stop in patch_segments]
    array_slices = []

    for start, stop in grid_segments:
        segment_size = max(abs(start), abs(stop))
        k = int(ceil(float(segment_size) / array_size) + (-1 if start >= 0 else 0))
        cell_mirrored = k % 2
        
        step = 1
        if start < 0:
            start = k * array_size + start
            stop = k * array_size + stop
        else:
            start = start - k * array_size
            stop = stop - k * array_size

        if cell_mirrored:
            start = array_size - start - 1
            stop = array_size - stop - 1
            step = -1

        if stop < 0:
            stop = None

        array_slices.append(slice(start, stop, step))
        
    return array_slices, patch_slices


def get_ndpatch(array, shape, index):
    """Returns a copy of the array region (patch) with given corner index and shape.
    
    Arguments:
        array (ndarray) -- array from which the patch is extracted
        shape (tuple) -- shape/size of the patch
        index (tuple) -- corner index of the patch
    
    Returns:
        (ndarray) -- A copy of the array region (patch)
    """
    if isinstance(shape, int):
        shape = (shape,)

    if not _is_sequence_of_integers(shape):
        raise TypeError("'shape' should be a sequence of integer values. | shape={}".format(shape))
    
    if isinstance(index, int):
        index = (index,)

    if not _is_sequence_of_integers(index):
        print (isinstance(index, collections.abc.Sequence))
        raise TypeError("'index' should be a sequence of integer values. | index={}".format(index))
    
    if len(shape) != len(array.shape):
        raise TypeError("The lengths of 'array.shape' and 'shape' are not the same. | array.shape={}, shape={}".format(array.shape, shape))

    if len(shape) != len(index):
        raise TypeError("The lengths of 'shape' and 'index' are not the same. | shape={}, index={}".format(shape, index))

    patch_shape = np.array(shape, dtype=np.uint32)
    index = np.array(index, dtype=np.int32)

    array_slices, patch_slices = [], []
    for array_dim_size, patch_dim_size, dim_index in zip(array.shape, patch_shape, index):
        grid_dim_segments, patch_dim_segments = find_segments(array_dim_size, patch_dim_size, dim_index)
        array_dim_slices, patch_dim_slices = _segments2slices(array_dim_size, grid_dim_segments, patch_dim_segments)
        array_slices.append(array_dim_slices)
        patch_slices.append(patch_dim_slices)

    array_rois = list(itertools.product(*array_slices))
    patch_rois = list(itertools.product(*patch_slices))

    patch = np.zeros(patch_shape, dtype=array.dtype)
    for aroi, proi in zip(array_rois, patch_rois):
        patch[proi] = array[aroi]
    
    return patch


def get_random_ndpatch(array, shape):
    """Returns a copy of random array region (patch) with given shape.
    
    Arguments:
        array (ndarray) -- array from which the patch is extracted
        shape (tuple) -- shape/size of the patch
    
    Returns:
        (ndarray) -- A copy of the array region (patch)
    """

    index = get_random_patch_index(array.shape, shape)
    return get_ndpatch(array, shape, index)
