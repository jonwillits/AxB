import torch
import time
import sys
import numpy as np
import pyprind

from src import config


class RNN:
    def __init__(self, corpus_list,
                 rnn_type = config.RNN.rnn_type,
                 hidden_size = config.RNN.hidden_size,
                 epochs = config.RNN.epochs,
                 learning_rate = config.RNN.learning_rate,
                 weight_init = config.RNN.weight_init,
                 seed = config.General.seed,
                 num_eval_steps = config.RNN.num_eval_steps,
                 bptt = config.RNN.bptt,
                 batch_size = config.RNN.batch_size,
                 num_layers = config.RNN.num_layers,
                 dropout_prob = config.RNN.dropout_prob,
                 grad_clip = config.RNN.grad_clip):

        self.corpus_list = corpus_list
        self.generate_master_vocab()

        self.pad_id = 0  # TODO this is the id for the pad symbol - make sure this is correct
        self.input_size = self.master_vocab_size
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.weight_init = weight_init
        self.seed = seed
        self.num_eval_steps = num_eval_steps
        self.bptt = bptt
        self.batch_size = batch_size

        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.grad_clip = grad_clip

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = TorchRNN(self.rnn_type, self.num_layers, self.input_size, self.hidden_size, self.batch_size,
                              self.weight_init)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate[0])
        self.model.cuda()

    def generate_master_vocab(self):
        self.master_vocab_list = []
        self.master_vocab_index_dict = {}

        index_counter = 0
        for corpus in self.corpus_list:
            for token in corpus.vocab_list:
                if token not in self.master_vocab_index_dict:
                    self.master_vocab_index_dict[token] = index_counter
                    self.master_vocab_list.append(token)
                    index_counter += 1
        self.master_vocab_size = len(self.master_vocab_list)

    def gen_windows(self, seq):
        # yield num_steps matrices where each matrix contains windows of size num_steps
        remainder = len(seq) % self.bptt
        for i in range(self.bptt):
            windows = np.roll(seq, -i)  # rightward
            windows = windows[:-remainder]

            # TODO
            print('windows')
            print(windows)

            x = np.reshape(windows, (-1, self.bptt))
            y = np.roll(x, -1)
            yield i, x, y

    def gen_batches(self, seqs, batch_size, verbose):
        for seq in seqs:
            # pad
            seq = [self.pad_id] * (self.bptt - 1) + seq

            # TODO debug
            print('===========================================')
            print(seq)
            print()

            for windows_id, x, y in self.gen_windows(seq):  # more memory efficient


                # TODO debug
                print('x')
                print(x)
                print('y')
                print(y)

                # exclude some rows to split x and y evenly by batch size
                shape0 = len(x)
                num_excluded = shape0 % batch_size
                if num_excluded > 0:  # in case mb_size = 1
                    x = x[:-num_excluded]
                    y = y[:-num_excluded]
                shape0_adj = shape0 - num_excluded
                # split into batches
                num_batches = shape0_adj // batch_size
                if verbose:
                    print('Excluding {} windows due to fixed batch size'.format(num_excluded))
                    print('{}/{} Generating {:,} batches with size {}...'.format(
                        windows_id + 1, self.bptt, num_batches, batch_size))
                for x_b, y_b in zip(np.vsplit(x, num_batches),
                                    np.vsplit(y, num_batches)):
                    yield x_b, y_b[:, -1]

    def train_epoch(self, seqs, lr, verbose):
        start_time = time.time()
        self.model.train()
        self.model.batch_size = self.batch_size
        # shuffle
        np.random.shuffle(seqs)

        print(seqs)

        # create batches within docs only
        for batch_id, (x_b, y_b) in enumerate(self.gen_batches(seqs, self.batch_size, verbose)):
            # forward step
            inputs = torch.cuda.LongTensor(x_b.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y_b)
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
            # console
            if batch_id % self.num_eval_steps == 0 and verbose:
                xent_error = loss.item()
                pp = np.exp(xent_error)
                secs = time.time() - start_time
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(
                    batch_id, pp, secs))

    def train(self, verbose=False):
        print('Training...')
        # train loop
        lr = self.learning_rate[0]  # initial
        decay = self.learning_rate[1]
        num_epochs_without_decay = self.learning_rate[2]
        pbar = pyprind.ProgBar(self.epochs, stream=sys.stdout)
        for epoch in range(self.epochs):
            if verbose:
                print('Starting epoch {} with lr={}'.format(epoch, lr))
            lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
            lr = lr * lr_decay  # decay lr if it is time
            self.train_epoch(self.corpus.index_sequence_list, lr, verbose)
            if verbose:
                print('\nTraining perplexity at epoch {}: {:8.2f}'.format(
                    epoch, self.calc_pp(self.corpus.index_sequence_list, verbose)))  # TODO do for categories separately
            else:
                pbar.update()
        wx_weights = self.model.wx.weight.detach().cpu().numpy()

    def calc_pp(self, seqs, verbose):
        if verbose:
            print('Calculating perplexity...')
        self.model.eval()
        self.model.batch_size = 1  # TODO probably better to do on CPU - or find batch size that excludes least samples
        errors = 0
        batch_id = 0
        token_ids = np.hstack(seqs)
        num_windows = len(token_ids)
        pbar = pyprind.ProgBar(num_windows, stream=sys.stdout)
        for batch_id, (x_b, y_b) in enumerate(self.gen_batches(seqs, self.batch_size, verbose)):
            pbar.update()
            inputs = torch.cuda.LongTensor(x_b.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y_b)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            logits = self.model(inputs, hidden)
            #
            self.optimizer.zero_grad()  # sets all gradients to zero
            loss = self.criterion(logits.unsqueeze_(0), targets)  # need to add dimension due to mb_size = 1
            errors += loss.item()
        res = np.exp(errors / batch_id + 1)
        return res


class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, hidden_size, batch_size, init_range):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
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