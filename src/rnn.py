import torch
import time
import numpy as np
import pandas as pd
from cytoolz import itertoolz
from src import config


class RNN:
    def __init__(self,
                 master_vocab,
                 rnn_type=config.RNN.rnn_type,
                 hidden_size=config.RNN.hidden_size,
                 epochs=config.RNN.epochs,
                 learning_rate=config.RNN.learning_rate,
                 optimization=config.RNN.optimization,
                 init_range=config.RNN.init_range,
                 seed=config.General.seed,
                 bptt=config.RNN.bptt,
                 num_seqs_in_batch=config.RNN.num_seqs_in_batch,
                 shuffle_seqs=config.RNN.shuffle_seqs,
                 num_layers=config.RNN.num_layers,
                 dropout_prob=config.RNN.dropout_prob):
        # input
        self.master_vocab = master_vocab
        self.input_size = master_vocab.master_vocab_size
        self.pad_id = 0
        # rnn
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.seed = seed
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.bptt = bptt
        self.num_seqs_in_batch = num_seqs_in_batch
        self.shuffle_seqs = shuffle_seqs
        self.learning_rate = learning_rate
        self.optimization = optimization
        self.init_range = init_range
        #
        self.model = TorchRNN(self.rnn_type, self.num_layers, self.input_size, self.hidden_size, self.init_range)
        self.criterion = torch.nn.CrossEntropyLoss()
        if self.optimization == 'adagrad':
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate)
        elif self.optimization == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        else:
            raise AttributeError('Invalid arg to "optimizer"')

    def to_windows(self, seq):
        padded = [self.pad_id] * self.bptt + seq
        bptt_p1 = self.bptt + 1
        seq_len = len(seq)
        windows = [padded[i: i + bptt_p1] for i in range(seq_len)]
        return windows

    def batch_windows(self, seqs, num_seqs_in_batch=None):
        """
        a batch, by default, contains all windows in a single sequence.
        setting "num_seqs_in_batch" larger than 1, will include all windows in "num_seqs_in_batch" sequences
        """
        if num_seqs_in_batch is None:
            num_seqs_in_batch = self.num_seqs_in_batch
        windowed_seqs = [self.to_windows(seq) for seq in seqs]
        if len(windowed_seqs) % num_seqs_in_batch != 0:
            raise RuntimeError('Set number of sequences in batch to factor of number of sequences {}.'.format(
                len(seqs)))
        for windowed_seqs_partition in itertoolz.partition_all(num_seqs_in_batch, windowed_seqs):
            batch = np.vstack(windowed_seqs_partition)
            yield batch

    def train_epoch(self, seqs, verbose, num_eval_steps):
        """
        each batch contains all windows in a sequence.
        hidden states are never saved. not across windows, and not across sequences.
        this guarantees that train updates are never based on any previous leftover information - no cheating.
        """
        start_time = time.time()
        self.model.train()
        if self.shuffle_seqs:
            np.random.shuffle(seqs)

        for step, windows in enumerate(self.batch_windows(seqs)):
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
            loss.backward()
            self.optimizer.step()

            # console
            if step % num_eval_steps == 0 and verbose:
                batch_pp = np.exp(loss.item())
                secs = time.time() - start_time
                print(x)
                print(y)
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(
                    step, batch_pp, secs))

    def calc_accuracies(self, seqs, corpus):
        all_windows = np.vstack([self.to_windows(seq) for seq in seqs])
        y = all_windows[:, -1]
        logits = self.calc_logits(seqs)

        # init counts
        correct_count = [0, 0, 0, 0]
        n = [0, 0, 0, 0]
        acc = [0, 0, 0, 0]

        for i in range(len(y)):
            correct_index = y[i]
            guess_index = np.argmax(logits[i])
            correct_label = self.master_vocab.master_vocab_list[correct_index]
            guess_label = self.master_vocab.master_vocab_list[guess_index]

            if correct_index == guess_index:
                correct = 1
            else:
                correct = 0

            if correct_label in corpus.category_item_lists_dict['A']:
                category = 0
            elif correct_label in corpus.category_item_lists_dict['x']:
                category = 1
            elif correct_label in corpus.category_item_lists_dict['B']:
                category = 2
            elif correct_label in corpus.category_item_lists_dict['.']:
                category = 3
            else:
                raise AttributeError('Invalid arg to "category"')

            correct_count[category] += correct
            n[category] += 1

            for j in range(len(n)):
                if n[j] == 0:
                    acc[j] = -1
                else:
                    acc[j] = float(correct_count[j])/n[j]

        return acc

    def calc_seqs_pp(self, seqs):
        self.model.eval()  # protects from dropout
        all_windows = np.vstack([self.to_windows(seq) for seq in seqs])
        self.model.batch_size = len(all_windows)

        # prepare inputs
        x = all_windows[:, :-1]
        y = all_windows[:, -1]
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
        targets = torch.LongTensor(y)

        # forward pass
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden)
        self.optimizer.zero_grad()  # sets all gradients to zero
        loss = self.criterion(logits, targets).item()
        res = np.exp(loss)
        return res

    def calc_logits(self, seqs):
        self.model.eval()  # protects from dropout
        all_windows = np.vstack([self.to_windows(seq) for seq in seqs])
        self.model.batch_size = len(all_windows)

        # prepare inputs
        x = all_windows[:, :-1]
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]

        # forward pass
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden).detach().numpy()
        return logits


class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, hidden_size, init_range):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = None  # is set dynamically
        self.init_range = init_range
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
