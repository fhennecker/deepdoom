# Replay memory minimum and maximum size
MIN_MEM_SIZE, MAX_MEM_SIZE = 2400, 80000

# Batch size for NN ingestion
BATCH_SIZE = 10

# Sequence length for NN ingestion
SEQUENCE_LENGTH = 8

# Number of training steps
TRAINING_STEPS = 1000

# Maximum number of cores to use
MAX_CPUS = 32

# Number of possible actions
N_ACTIONS = 2

# Learning rate for tensorflow optimizers
LEARNING_RATE = 0.0001

try:
    from local_config import *
except ImportError:
    pass
