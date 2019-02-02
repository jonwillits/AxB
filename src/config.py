

class General:
    seed = None


class AxB:
    # the basic parameters of the AxB grammar
    ab_types = 2
    x_types = 2
    min_distance = 1
    max_distance = 1
    punct = True

    # if sample is False, all combos will be generated
    # if sample is true, num_sequences will be randomly sampled from full set
    sample = False
    num_sequences = 10
    noise = 0.1


class RNN:
    rnn_type = 'srn'
    num_layers = 1
    dropout_prob = 0.0
    grad_clip = None
    bptt = 3  # if larger than length of seq, pad_id is used to buffer left side of windows
    num_seqs_in_batch = 1
    shuffle_seqs = True
    hidden_size = 10
    epochs = 200
    learning_rate = [0.1, 0.01, 10]
    init_range = 0.001
    optimization = 'adagrad'