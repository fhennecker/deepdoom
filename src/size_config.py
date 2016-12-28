MIN_MEM_SIZE, MAX_MEM_SIZE = 900, 10000
TRAINING_STEPS = 100

try:
    from local_size_config import *
except ImportError:
    pass
