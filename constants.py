HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10
NUM_DATA_FILES = 5

# We use a weight decay of 0.0002, which performs better than the 0.0001 that
# was originally suggested.
WEIGHT_DECAY = 2e-4
MOMENTUM = 0.9

NUM_IMAGES = {
    'train': 50000,
    'validation': 10000,
}