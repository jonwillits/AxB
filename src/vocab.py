
class Vocab:
    def __init__(self, *corpus_list):
        self.corpus_list = [corpus for corpus in corpus_list]
        self.master_vocab_index_dict, self.master_vocab_list = self.generate_master_vocab()
        self.master_vocab_size = len(self.master_vocab_list)

    def generate_master_vocab(self):
        master_vocab_list = []
        master_vocab_index_dict = {}
        index_counter = 0
        for corpus in self.corpus_list:
            for token in corpus.vocab_list:
                if token not in master_vocab_index_dict:
                    master_vocab_index_dict[token] = index_counter
                    master_vocab_list.append(token)
                    index_counter += 1
        return master_vocab_index_dict, master_vocab_list

    def generate_index_sequences(self, corpus):
        index_sequence_list = []
        for sequence in corpus.sequence_sample:
            new_sequence = []
            for token in sequence:
                new_sequence.append(self.master_vocab_index_dict[token])
            index_sequence_list.append(new_sequence)
        return index_sequence_list
