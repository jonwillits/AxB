import numpy as np
import random


def calc_min_seqs_pp(input_params, num_types):
    denom = input_params.ab_cat_size + input_params.train_x_cat_size + int(input_params.punctuation) + 1  # + 1 for B
    min_seqs_pp = num_types / denom
    return min_seqs_pp


def calc_max_cat_pp(input_params, num_sequences, num_items):
    # max_cat_pp is is cat_pp for punctuation
    # because the punctuation category only consists of 1 item and therefore has the least probability mass
    avg_window_size = int(input_params.punctuation) + 2 + np.mean([input_params.max_distance,
                                                                   input_params.min_distance])
    num_windows = avg_window_size * num_sequences
    max_cat_pp = np.exp((num_sequences / num_windows) * -np.log(1 / num_items))
    return max_cat_pp


def calc_max_item_pp_axb(input_params, cat):
    # max_item_pp is item_pp for the category which has largest set size
    # because least amount of probability mass is initially devoted to the correct item
    num_items_in_cat = input_params.ab_cat_size if cat in ['A', 'B'] else input_params.train_x_cat_size
    max_item_pp = np.exp(1 * -np.log(1 / num_items_in_cat))
    return max_item_pp


def calc_max_item_pp_marcus(input_params):
    num_items_in_cat = input_params.ab_cat_size
    max_item_pp = np.exp(1 * -np.log(1 / num_items_in_cat))
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


def make_random_sequences(items, seq_size, num_sequences, exclude_pattern=None):
    items = [i for i in items if i != '.']
    res = []
    while not len(res) >= num_sequences:
        random_sequence = np.random.choice(items, size=seq_size, replace=True).tolist()
        if is_pattern_match(random_sequence, exclude_pattern):
            print('Excluded:', random_sequence)
            continue
        res.append(random_sequence)
    return res


def is_pattern_match(seq, pattern):
    if pattern == 'aab':
        if seq[0] == seq[1] != seq[2]:
            return True
        else:
            return False
    else:
        raise NotImplementedError