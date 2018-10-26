from src import corpus
from src import rnn
from src import utils

verbose = True
ab_types = 2
x_train_types = 3
x_test_types = 4
max_distance = 1
min_distance = 1

corpus_list = []
corpus_list.append(corpus.AxbCorpus(ab_types, x_train_types, max_distance, min_distance))
corpus_list.append(corpus.AxbCorpus(ab_types, x_test_types, max_distance, min_distance))

srn = rnn.RNN(corpus_list)
srn.train(0, verbose)

