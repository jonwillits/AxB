import os
from pathlib import Path

class General:
    seed = None

class AxB:
    # the basic parameters of the AxB grammar
    ab_types = 2
    x_types = 3
    min_distance = 1
    max_distance = 1
    punct = True

    # if sample is False, all combos will be generated
    # if sample is true, num_sequences will be randomly sampled from full set
    sample = False
    num_sequences = 10
    noise = 0.1

class RNN:
    hidden_size = 10
    epochs = 1
    learning_rate = [0.01, 0.01, 10]
    weight_init = 0.3