

class Verbosity:
    softmax = True
    summary = False
    cat_pp = False
    item_pp = False
    seqs_pp = False


class Eval:
    max_distance = 3  # for 'test' sequences
    skip_novel = True


class Marcus:
    # the basic parameters of the Marcus corpus
    pattern = 'abb'  # chose from ['abb', 'aab', 'aba']
    train_ab_cat_size = 3
    test_ab_cat_size = 6
    punctuation_at_start = True  # punctuation is at start of sequence
    punctuation = False  # punctuation is at end of sequence
    sample_size = None  # if None, full set is used, else sample from full set


class Axb:
    # the basic parameters of the AxB grammar
    ab_cat_size = 2
    train_x_cat_size = 6
    test_x_cat_size = 6
    min_distance = 1
    max_distance = 2
    punctuation = False   # punctuation is at end of sequence
    sample_size = None  # if None, full set is used, else sample from full set


class RNN:
    rnn_type = 'srn'
    num_layers = 1
    dropout_prob = 0.0
    bptt = 3  # if larger than length of seq, pad_id is used to buffer left side of windows
    num_seqs_in_batch = 1
    shuffle_seqs = True
    hidden_size = 8
    num_epochs = 100
    learning_rate = 0.25
    init_range = 0.001
    optimization = 'adagrad'
    no_batching = True