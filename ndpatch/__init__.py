__version__ = '0.0.1'

try:
    from ndpatch.ndpatch import *
except ImportError:
    from ndpatch import *