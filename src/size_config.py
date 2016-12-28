MIN_MEM_SIZE, MAX_MEM_SIZE = 1000, 10000
TRAINING_STEPS = 1000

try:
    from local_size_config import *
except ImportError:
    pass
