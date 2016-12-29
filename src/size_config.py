MIN_MEM_SIZE, MAX_MEM_SIZE = 2400, 80000
TRAINING_STEPS = 1000

try:
    from local_size_config import *
except ImportError:
    pass
