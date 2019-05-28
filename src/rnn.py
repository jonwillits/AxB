import torch
import time
import numpy as np
from cytoolz import itertoolz
from src import config


class RNN:
    def __init__(self, input_size, params):
        self.input_size = input_size
        self.pad_id = 0
        self.params = params
        #
        self.model = TorchRNN(self.params.rnn_type, self.params.num_layers, self.input_size,
                              self.params.hidden_size, self.params.init_range, self.params.dropout_prob)
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.params.optimization == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.params.learning_rate)
        elif self.params.optimization == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.params.learning_rate)
        else:
            raise AttributeError('Invalid arg to "optimizer"')

    def to_windows(self, seq):
        padded = [self.pad_id] * self.params.bptt + seq
        bptt_p1 = self.params.bptt + 1
        seq_len = len(seq)
        windows = [padded[i: i + bptt_p1] for i in range(seq_len)]
        return windows

    def batch_windows(self, seqs, num_seqs_in_batch=None):
        """
        a batch, by default, contains all windows in a single sequence.
        setting "num_seqs_in_batch" larger than 1, will include all windows in "num_seqs_in_batch" sequences
        """
        if num_seqs_in_batch is None:
            num_seqs_in_batch = self.params.num_seqs_in_batch
        all_windows = [self.to_windows(seq) for seq in seqs]
        if len(all_windows) % num_seqs_in_batch != 0:
            raise RuntimeError('Set number of sequences in batch to factor of number of sequences {}.'.format(
                len(seqs)))
        for windows_in_batch in itertoolz.partition_all(num_seqs_in_batch, all_windows):
            yield np.vstack(windows_in_batch)

    def train_epoch(self, seqs, train=True):
        """
        each batch contains all windows in a sequence.
        hidden states are never saved. not across windows, and not across sequences.
        this guarantees that train updates are never based on any previous leftover information - no cheating.
        """
        self.model.train()
        if self.params.shuffle_seqs:
            np.random.shuffle(seqs)

        total_pp = 0
        num_batches = 0
        for windows in self.batch_windows(seqs):
            self.model.batch_size = len(windows)  # dynamic batch size
            x = windows[:, :-1]
            y = windows[:, -1]

            # forward step
            inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
            targets = torch.LongTensor(y)
            hidden = self.model.init_hidden()  # must happen, because batch size changes from seq to seq
            logits = self.model(inputs, hidden)

            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits, targets)
            if train:  # otherwise just return perplexity (useful to know before any training has begun)
                loss.backward()
                self.optimizer.step()

            # perplexity
            total_pp += np.exp(loss.item())
            num_batches += 1

        seqs_pp = total_pp / num_batches
        return seqs_pp

    def to_x_and_y(self, seqs):
        all_windows = np.vstack([self.to_windows(seq) for seq in seqs])
        x = all_windows[:, :-1]
        y = all_windows[:, -1]
        return x, y

    def calc_seqs_pp(self, seqs):
        x, y = self.to_x_and_y(seqs)
        self.model.eval()  # protects from dropout
        self.model.batch_size = len(x)
        # forward pass
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
        targets = torch.LongTensor(y)
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden)
        self.optimizer.zero_grad()  # sets all gradients to zero
        loss = self.criterion(logits, targets).item()
        res = np.exp(loss)
        return res

    def calc_logits(self, seqs):
        x, y = self.to_x_and_y(seqs)
        self.model.eval()  # protects from dropout
        self.model.batch_size = len(x)
        # forward pass
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden).detach().numpy()
        return logits


class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, hidden_size, init_range, dropout_prob):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = None  # is set dynamically
        self.init_range = init_range
        self.dropout_prob = dropout_prob
        #
        self.wx = torch.nn.Embedding(input_size, self.hidden_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=self.hidden_size,
                             hidden_size=self.hidden_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.wy = torch.nn.Linear(in_features=self.hidden_size,
                                  out_features=input_size)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-self.init_range, self.init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.init_range, self.init_range)

    def init_hidden(self, verbose=False):
        if verbose:
            print('Initializing hidden weights with size [{}, {}, {}]'.format(
                self.num_layers, self.batch_size, self.hidden_size))
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            res = (torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.hidden_size).zero_()),
                   torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.hidden_size).zero_()))
        else:
            res = torch.autograd.Variable(weight.new(self.num_layers,
                                                     self.batch_size,
                                                     self.hidden_size).zero_())
        return res

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(final_outputs)
        return logits
