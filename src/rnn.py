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
                 init_range=config.RNN.init_range,
                 seed=config.General.seed,
                 num_eval_steps=config.RNN.num_eval_steps,
                 bptt=config.RNN.bptt,
                 num_seqs_in_batch=config.RNN.num_seqs_in_batch,
                 num_layers=config.RNN.num_layers,
                 dropout_prob=config.RNN.dropout_prob,
                 grad_clip=config.RNN.grad_clip):
        # input
        self.master_vocab = master_vocab
        self.input_size = master_vocab.master_vocab_size
        self.pad_id = 0
        # rnn
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.seed = seed
        self.num_eval_steps = num_eval_steps
        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.grad_clip = grad_clip
        self.bptt = bptt
        self.num_seqs_in_batch = num_seqs_in_batch
        self.learning_rate = learning_rate
        self.model = TorchRNN(self.rnn_type, self.num_layers, self.input_size, self.hidden_size, init_range)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate[0])

    def train(self, seqs, corpus, verbose=False):
        print('{:13s} {:10s}{:10s}{:10s}{:10s}{:10s}'.format('Epoch', 'Perp', 'A', 'x', 'B', '.'))
        lr = self.learning_rate[0]  # initial
        decay = self.learning_rate[1]
        num_epochs_without_decay = self.learning_rate[2]
        for epoch in range(self.epochs):
            lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
            lr = lr * lr_decay  # decay lr if it is time
            self.train_epoch(seqs, lr, verbose)

            pp = self.calc_pp(seqs)
            accs = self.calc_accuracy(seqs, corpus)
            print('{:8}: {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}  {:8.2f}'.format(epoch, pp, accs[0], accs[1], accs[2], accs[3]))

        y, logits = self.get_logits(seqs)
        self.save_logits_to_disk(y, logits)

    def retrieve_wx_for_analysis(self):
        wx_weights = self.model.wx.weight.detach().cpu().numpy()  # if stored on gpu
        return wx_weights

    def to_windows(self, seq):
        padded = [self.pad_id] * self.bptt + seq
        bptt_p1 = self.bptt + 1
        seq_len = len(seq)
        windows = [padded[i: i + bptt_p1] for i in range(seq_len)]
        return windows

    def gen_batches(self, seqs, num_seqs_in_batch=None):
        if num_seqs_in_batch is None:
            num_seqs_in_batch = self.num_seqs_in_batch
        windowed_seqs = [self.to_windows(seq) for seq in seqs]
        if len(windowed_seqs) % num_seqs_in_batch != 0:
            raise RuntimeError('Set number of sequences in batch to factor of number of sequences.')
        for windowed_seqs_partition in itertoolz.partition_all(num_seqs_in_batch, windowed_seqs):
            batch = np.vstack(windowed_seqs_partition)
            yield batch

    def train_epoch(self, seqs, lr, verbose):
        start_time = time.time()
        self.model.train()
        np.random.shuffle(seqs)
        # a batch contains num_docs_in_batch sequences (each sequence consists of multiple windows)
        for batch_id, batch in enumerate(self.gen_batches(seqs, self.num_seqs_in_batch)):
            self.model.batch_size = len(batch)  # dynamic batch size
            x = batch[:, :-1]
            y = batch[:, -1]
            if verbose:
                print(x)
                print(y)
            # forward step
            inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
            targets = torch.LongTensor(y)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            logits = self.model(inputs, hidden)
            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits, targets)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                for p in self.model.parameters():
                    p.data.add_(-lr, p.grad.data)
            else:
                self.optimizer.step()
            #console
            if batch_id % self.num_eval_steps == 0 and verbose:
                xent_error = loss.item()
                pp = np.exp(xent_error)
                secs = time.time() - start_time
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(
                    batch_id, pp, secs))

    def calc_accuracy(self, seqs, corpus):
        y, logits = self.get_logits(seqs)
        correct_count = [0,0,0,0]
        n = [0,0,0,0]
        acc = [0,0,0,0]

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
            if correct_label in corpus.category_item_lists_dict['x']:
                category = 1
            if correct_label in corpus.category_item_lists_dict['B']:
                category = 2
            if correct_label in corpus.category_item_lists_dict['.']:
                category = 3

            correct_count[category] += correct
            n[category] += 1

            for j in range(len(n)):
                if n[j] == 0:
                    acc[j] = -1
                else:
                    acc[j] = float(correct_count[j])/n[j]

        return acc

    def calc_pp(self, seqs):
        #print('Calculating perplexity...')
        self.model.eval()  # protects from dropout
        batch = np.vstack([self.to_windows(seq) for seq in seqs])  # batch contains all seqs
        self.model.batch_size = len(batch)
        x = batch[:, :-1]
        y = batch[:, -1]
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
        targets = torch.LongTensor(y)
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden)
        #

        self.optimizer.zero_grad()  # sets all gradients to zero
        loss = self.criterion(logits, targets)
        errors = loss.item()
        res = np.exp(errors / len(seqs))
        return res

    def get_logits(self, seqs):
        self.model.eval()  # protects from dropout
        batch = np.vstack([self.to_windows(seq) for seq in seqs])  # batch contains all seqs
        self.model.batch_size = len(batch)
        x = batch[:, :-1]
        y = batch[:, -1]
        inputs = torch.LongTensor(x.T)  # requires [num_steps, mb_size]
        hidden = self.model.init_hidden()  # this must be here to re-init graph
        logits = self.model(inputs, hidden)
        data = logits.detach().numpy()

        return y, data

    def save_logits_to_disk(self, y, logits):  # TODO test
        df = pd.DataFrame(index=y, data=logits)
        df.index.name = 'y'
        df.to_csv('logits.csv')

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

    def init_hidden(self):
        #print('Initializing hidden weights with size [{}, {}, {}]'.format(
            #self.num_layers, self.batch_size, self.hidden_size))
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
