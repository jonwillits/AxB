
class Vocab:
    def __init__(self, *corpus_list):
        self.corpus_list = [corpus for corpus in corpus_list]
        self.types = self.make_types()
        self.type2id = {t: n for n, t in enumerate(self.types)}
        #
        assert 'B1' == self.types[0]  # type-perplexity evaluation assumes B types come first in output
        assert self.type2id['B1'] == 0

        self.type2id['PAD'] = len(self.types)
        self.types.append('PAD')
        print('Initializing vocab with punctuation id={}'.format(len(self.types)))
        for k, v in sorted(self.type2id.items(), key=lambda i: i[1]):
            print(k, v)

        self.num_types = len(self.types)

    def make_types(self):
        res = []
        for corpus in self.corpus_list:
            for t in corpus.types:
                if t not in res:
                    res.append(t)
        return res

    def generate_index_sequences(self, corpus):
        res = []
        for sequence in corpus.sequence_sample:
            new_sequence = []
            for token in sequence:
                new_sequence.append(self.type2id[token])
            res.append(new_sequence)
        return res
