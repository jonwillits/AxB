from src import corpus
from src import rnn
from src import vocab

ab_types = 2
x_train_types = 2
x_test_types = 2
max_distance = 2
min_distance = 1


train_corpus = corpus.AxbCorpus(ab_types, x_train_types, max_distance, min_distance)
test_corpus = corpus.AxbCorpus(ab_types, x_test_types, max_distance, min_distance)

master_vocab = vocab.Vocab(train_corpus, test_corpus)
train_seqs = master_vocab.generate_index_sequences(train_corpus)
for seq in train_seqs:
    print(seq)

srn = rnn.RNN(master_vocab)
srn.train(train_seqs, train_corpus, verbose=False)

