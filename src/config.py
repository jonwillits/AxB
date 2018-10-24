import os
from pathlib import Path

class Dirs:
    corpora = Path(os.getenv('CORPORA_DIR', Path(__file__).parent.parent / 'corpora'))

class AxB:
    # the basic parameters of the AxB grammar
    ab_types = 2
    x_types = 3
    min_distance = 3
    max_distance = 3
    punct = False

    # if sample is False, all combos will be generated
    # if sample is true, num_sequences will be randomly sampled from full set
    num_sequences = 16
    sample = False
    noise = 0
    seed = None