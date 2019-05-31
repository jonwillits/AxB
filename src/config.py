

class General:
    num_reps = 1
    cat_pp_verbose = False
    type_pp_verbose = False
    seqs_pp_verbose = False


class Input:
    # the basic parameters of the AxB grammar
    num_ab_types = 2
    num_x_train_types = 2
    num_x_test_types = 4
    min_distance = 1
    max_distance = 2
    punct = True
    #
    sample_size = None  # if None, full set is used, else sample from full set
    num_sequences = 10


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