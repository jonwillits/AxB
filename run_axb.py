import numpy as np
import pandas as pd

from src.corpus import AxbCorpus
from src.rnn import RNN
from src.vocab import Vocab

ab_types = 2
x_train_types = 4
x_test_types = 8
max_distance = 1
min_distance = 1

# corpora
train_corpus = AxbCorpus(ab_types, x_train_types, max_distance, min_distance)
test_corpus = AxbCorpus(ab_types, x_test_types, max_distance, min_distance)

# train sequences
master_vocab = Vocab(train_corpus, test_corpus)
train_seqs = master_vocab.generate_index_sequences(train_corpus)
for seq in train_seqs:
    print(seq)

# train SRN
srn = RNN(master_vocab)
srn.train(train_seqs, train_corpus, verbose=False)

# evaluation
all_windows = np.vstack([srn.to_windows(seq) for seq in train_seqs])
y = all_windows[:, -1]
logits = srn.calc_logits(train_seqs)

# save to disk
df = pd.DataFrame(index=y, data=logits)
df.index.name = 'y'
df.to_csv('logits.csv')