
class Vocab:
    def __init__(self, *corpus_list):
        self.corpus_list = [corpus for corpus in corpus_list]
        self.type2id, self.types = self.generate_master_vocab()
        self.master_vocab_size = len(self.types)

    def generate_master_vocab(self):
        types = []
        type2id = {}
        index_counter = 0
        for corpus in self.corpus_list:
            for token in corpus.vocab_list:
                if token not in type2id:
                    type2id[token] = index_counter
                    types.append(token)
                    index_counter += 1
        return type2id, types

    def generate_index_sequences(self, corpus):
        index_sequence_list = []
        for sequence in corpus.sequence_sample:
            new_sequence = []
            for token in sequence:
                new_sequence.append(self.type2id[token])
            index_sequence_list.append(new_sequence)
        return index_sequence_list
