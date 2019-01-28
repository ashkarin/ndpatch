.. image:: static/ndpatch.svg
    :height: 120
    :align: center
    
-----------

.. image:: https://travis-ci.org/ashkarin/ndpatch.svg?branch=master 
    :target: https://travis-ci.org/ashkarin/ndpatch


**NDPatch** is the package for extracting arbitrary regions from an N-dimensional numpy array assuming it mirrored infinitely.

Installation
------------

The easiest way to install the latest version is by using pip::

    $ pip install ndpatch

You may also use Git to clone the repository and install it manually::

    $ git clone https://github.com/ashkarin/ndpatch.git
    $ cd ndpatch
    $ python setup.py install

Usage
-----
To take a patch from the array:

.. code-block:: python

  import numpy as np
  import ndpatch
  array = np.arange(25).reshape((5,5))
  index = (1, 2)
  shape = (3, 3)
  patch = ndpatch.get_ndpatch(array, shape, index)
  # patch =
  # [[ 7,  8,  9],
  #  [12, 13, 14],
  #  [17, 18, 19]]

To take get a random patch index:

.. code-block:: python

  import numpy as np
  import ndpatch
  array_shape = (5, 5)
  index = ndpatch.get_random_patch_index(array_shape)

To extract random patches from the array:

.. code-block:: python

  import numpy as np
  import ndpatch
  npatches = 10
  patch_shape = (3, 3)
  array = np.arange(100).reshape((10,10))
  patches = [ndpatch.get_random_ndpatch(array, patch_shape) for _ in range(npatches)]

To split the 3D array on set of overlapping 3D patches and rebuild it back:

.. code-block:: python

  import numpy as np
  import ndpatch
  array = np.arange(0, 125).reshape((5,5,5))
  patch_shape = (4, 3, 3)
  overlap = 2
  indices = ndpatch.get_patches_indices(array.shape, patch_shape, overlap)
  patches = [ndpatch.get_ndpatch(array, patch_shape, index) for index in indices]
  reconstructed = ndpatch.reconstruct_from_patches(patches, indices, array.shape, default_value=0)
  # Validate
  equal = (reconstructed == array)
  assert (np.all(equal))
