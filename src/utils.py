import numpy as np


def calc_min_seqs_pp(input_params, num_types):
    denom = input_params.ab_cat_size + input_params.train_x_cat_size + int(input_params.punctuation) + 1  # + 1 for B
    min_seqs_pp = num_types / denom
    return min_seqs_pp


def calc_max_cat_pp(input_params, num_sequences, num_types):
    # max_cat_pp is is cat_pp for punctuation
    # because the punctuation category only consists of 1 item and therefore has the least probability mass
    avg_window_size = int(input_params.punctuation) + 2 + np.mean([input_params.max_distance,
                                                                      input_params.min_distance])
    num_windows = avg_window_size * num_sequences
    max_cat_pp = np.exp(-np.log(1 / num_types) * num_sequences / num_windows)
    return max_cat_pp


def calc_max_item_pp(input_params):
    # max_item_pp is item_pp for the category which has largest set size
    # because least amount of probability mass is initially devoted to the correct item
    #
    assert input_params.min_distance == input_params.max_distance  # only works with this constraint
    max_num_types = np.max([input_params.ab_cat_size,
                            input_params.train_x_cat_size])
    window_size = input_params.max_distance + 1 + 1 + 1
    max_item_pp = np.exp(-np.log(1 / max_num_types) * (1 / window_size))
    return max_item_pp


def print_params(params):
    print()
    print('===============================================')
    for k, v in sorted(params.__dict__.items()):
        if not k.startswith('__'):
            print('{}={}'.format(k, v))


def to_string(params):
    res = ''
    for k, v in sorted(params.__dict__.items()):
        if not k.startswith('__'):
            res += '{}={}\n'.format(k, v)
    return res