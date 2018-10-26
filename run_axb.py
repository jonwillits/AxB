from src import corpus
from src import rnn
from src import vocab

ab_types = 2
x_train_types = 3
x_test_types = 4
max_distance = 1
min_distance = 1


corpus_list = [
    corpus.AxbCorpus(ab_types, x_train_types, max_distance, min_distance),
    corpus.AxbCorpus(ab_types, x_test_types, max_distance, min_distance)
]
master_vocab = vocab.Vocab(corpus_list)

srn = rnn.RNN(master_vocab.master_vocab_size)
srn.train(master_vocab.generate_index_sequences(corpus_number=0))

