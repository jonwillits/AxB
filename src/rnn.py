import torch
from src import config
from random import shuffle
from torch.autograd import Variable
import numpy as np
import pylab as pl
import torch.nn.init as init
from src import utils

import time
import torch
import pyprind
import numpy as np
import sys

from src import config

class RNN:
    def __init__(self, corpus,
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


        self.corpus = corpus
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size # fix embed size to this
        self.epochs = epochs
        self.weight_init = weight_init #fix embed_init_range to this
        self.seed = seed
        self.num_eval_steps = num_eval_steps
        self.bptt = bptt # num_steps
        self.batch_size = batch_size

        self.dropout_prob = dropout_prob
        self.num_layers = num_layers
        self.grad_clip = grad_clip

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = TorchRNN(self.rnn_type, self.num_layers, self.hidden_size, self.batch_size,
                              self.weight_init)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate[0])  # TODO Adagrad
        self.model.cuda()

    def gen_windows(self, token_ids):
        # yield num_steps matrices where each matrix contains windows of size num_steps
        remainder = len(token_ids) % self.bptt
        for i in range(self.bptt):
            seq = np.roll(token_ids, i)  # rightward
            seq = seq[:-remainder]
            x = np.reshape(seq, (-1, self.bptt))
            y = np.roll(x, -1)
            yield i, x, y

    def gen_batches(self, token_ids, batch_size, verbose):
        batch_id_list = []
        x_b_list = []
        y_b_list = []

        ''' 
            your old code flattened the inputs, removing the document separation, before the list got to that function
            that hasnt been done here since i need the minibatches to be based on number of AxB strings (effectively,
            the number of documents, as each string is in here as a document in the way you're used to), not the number
            of tokens. so the batching needs to take the list of strings/documents, break them into self.minibatch_size 
            number of chunks, flatten that and then generate the windows. because of this re-ordering, I didnt think your
            fancy yield function would work as smoothly, so I converted it to lists. Also, that will be easier for
            instruction...
        
            This function also needs to pad with .'s (vocab_index=0) so we start the window at the first item,
            not with the first item whose window is BPTT long.
            This needs to happen each time the window is reset.
            So if this reset happens every batch, then this padding needs to happen at every batch
            If the window never resets, then this padding only needs to happen at the very beginning
        '''

        batch_id = 0
        for window_id, x, y in self.gen_windows(
                token_ids):  # more memory efficient not to create all windows in data
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
                    window_id + 1, self.num_steps, num_batches, batch_size))
            for x_b, y_b in zip(np.vsplit(x, num_batches),
                                np.vsplit(y, num_batches)):
                batch_id_list.append(batch_id)
                x_b_list.append(x_b)
                y_b_list.append(y_b)
                batch_id += 1
        return batch_id_list, x_b_list, y_b_list


    def calc_pp(self, numeric_docs, verbose):
        if verbose:
            print('Calculating perplexity...')
        self.model.eval()
        self.model.batch_size = 1  # TODO probably better to do on CPU - or find batch size that excludes least samples
        errors = 0
        batch_id = 0
        token_ids = np.hstack(numeric_docs)
        num_windows = len(token_ids)
        pbar = pyprind.ProgBar(num_windows, stream=sys.stdout)
        for batch_id, x_b, y_b in self.gen_batches(token_ids, self.model.batch_size, verbose):
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

    def train_epoch(self, numeric_docs, lr, verbose):
        start_time = time.time()
        self.model.train()
        self.model.batch_size = self.batch_size

        # shuffle
        if self.shuffle_per_epoch:
            np.random.shuffle(numeric_docs)

        '''
        see extensive comments in gen_batches()
        '''
        batch_id_list, x_b_list, x_y_list = self.gen_batches(numeric_docs, self.batch_size, verbose)

        for i in range(self.batch_size):
            batch_id = batch_id_list[i]
            x_b = x_b_list[i]
            y_b = y_b_list[i]

            # forward step
            inputs = torch.cuda.LongTensor(x_b.T)  # requires [num_steps, mb_size]
            targets = torch.cuda.LongTensor(y_b)
            hidden = self.model.init_hidden()  # this must be here to re-init graph
            logits = self.model(inputs, hidden)
            # backward step
            self.optimizer.zero_grad()  # sets all gradients to zero  # TODO why put this here?
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
                print("batch {:,} perplexity: {:8.2f} | seconds elapsed in epoch: {:,.0f} ".format(batch_id, pp,
                                                                                                   secs))

    def train(self, verbose=False):
        print('Training...')
        # train loop
        lr = self.learning_rate[0]  # initial
        decay = self.learning_rate[1]
        num_epochs_without_decay = self.learning_rate[2]
        pbar = pyprind.ProgBar(self.epochs, stream=sys.stdout)
        for epoch in range(self.epochs):
            if verbose:
                print('/Starting epoch {} with lr={}'.format(epoch, lr))
            lr_decay = decay ** max(epoch - num_epochs_without_decay, 0)
            lr = lr * lr_decay  # decay lr if it is time
            self.train_epoch(self.corpus.index_sequence_list, lr, verbose)
            if verbose:
                print('\nTraining perplexity at epoch {}: {:8.2f}'.format(
                    epoch, self.calc_pp(self.corpus.index_sequence_list, verbose)))
            else:
                pbar.update()
        wx_weights = self.model.wx.weight.detach().cpu().numpy()  # TODO is this the correct order of vocab?

class TorchRNN(torch.nn.Module):
    def __init__(self, rnn_type, num_layers, input_size, embed_size, batch_size, embed_init_range):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.embed_init_range = embed_init_range
        #
        self.wx = torch.nn.Embedding(input_size, self.embed_size)
        if self.rnn_type == 'lstm':
            self.cell = torch.nn.LSTM
        elif self.rnn_type == 'srn':
            self.cell = torch.nn.RNN
        else:
            raise AttributeError('Invalid arg to "rnn_type".')
        self.rnn = self.cell(input_size=self.embed_size,
                             hidden_size=self.embed_size,
                             num_layers=self.num_layers,
                             dropout=self.dropout_prob if self.num_layers > 1 else 0)
        self.wy = torch.nn.Linear(in_features=self.embed_size,
                                  out_features=input_size)
        self.init_weights()

    def init_weights(self):
        self.wx.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)
        self.wy.bias.data.fill_(0.0)
        self.wy.weight.data.uniform_(-self.embed_init_range, self.embed_init_range)

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.rnn_type == 'lstm':
            res = (torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.embed_size).zero_()),
                   torch.autograd.Variable(weight.new(self.num_layers,
                                                      self.batch_size,
                                                      self.embed_size).zero_()))
        else:
            res = torch.autograd.Variable(weight.new(self.num_layers,
                                                     self.batch_size,
                                                     self.embed_size).zero_())
        return res

    def forward(self, inputs, hidden):  # expects [num_steps, mb_size] tensor
        embeds = self.wx(inputs)
        outputs, hidden = self.rnn(embeds, hidden)  # this returns all time steps
        final_outputs = torch.squeeze(outputs[-1])
        logits = self.wy(final_outputs)
        return logits