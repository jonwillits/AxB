import numpy as np


def calc_min_seqs_pp(input_params, master_vocab_size):
    denom = input_params.num_ab_types + input_params.num_x_train_types + 1 + 1  # 1 for punct and B
    min_seqs_pp = master_vocab_size / denom
    return min_seqs_pp


def calc_max_cat_pp(input_params, num_sequences, master_vocab_size):
    # max_cat_pp is is cat_pp for punctuation
    # because the punctuation category only consists of 1 type and therefore has the least probability mass
    avg_window_size = (1 if input_params.punct else 0) + 2 + np.mean([input_params.max_distance,
                                                                      input_params.min_distance])
    num_windows = avg_window_size * num_sequences
    max_cat_pp = -np.log(1 / master_vocab_size) * num_sequences / num_windows
    return max_cat_pp


def calc_max_type_pp(input_params, num_sequences):
    # max_type_pp is type_pp for the category which has largest set size
    # because least amount of probability mass is initially devoted to the correct type
    max_num_types = np.max([input_params.num_ab_types,
                            input_params.num_x_test_types])
    avg_window_size = (1 if input_params.punct else 0) + 2 + np.mean([input_params.max_distance,
                                                                      input_params.min_distance])
    num_windows = avg_window_size * num_sequences
    max_type_pp = -np.log(1 / max_num_types) * num_sequences / num_windows
    return max_type_pp


def print_params(params):
    print()
    print('===============================================')
    for k, v in sorted(params.__dict__.items()):
        if not k.startswith('__'):
            print('{}={}'.format(k, v))