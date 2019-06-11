
class Vocab:
    def __init__(self, *corpus_list):
        self.corpus_list = [corpus for corpus in corpus_list]
        self.items = self.make_items()
        self.item2id = {item: n for n, item in enumerate(self.items)}

        self.item2id['PAD'] = len(self.items)
        self.items.append('PAD')
        print('Initializing vocab with pad_id={}'.format(self.item2id['PAD']))
        for k, v in sorted(self.item2id.items(), key=lambda i: i[1]):
            print(k, v)

    @property
    def num_items(self):
        return len(self.items)

    def make_items(self):
        res = []
        for corpus in self.corpus_list:
            for item in corpus.items:
                if item not in res:
                    res.append(item)
        return res

    def generate_index_sequences(self, corpus):
        res = []
        for sequence in corpus.sequence_sample:
            new_sequence = []
            for token in sequence:
                new_sequence.append(self.item2id[token])
            res.append(new_sequence)
        return res
