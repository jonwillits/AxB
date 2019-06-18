from abc import ABC
from collections.abc import Mapping


class Results(Mapping, ABC):
    def __init__(self, corpus):
        self.corpus = corpus
        self.name = corpus.name
        #
        self._results = {cat: {position: {'cat_pps':  [],
                                          'item_pps': []} for position in corpus.positions}
                         for cat in corpus.cats}

    def __getitem__(self, key):
        return self._results[key]

    def __iter__(self):
        return iter(self._results)

    def __len__(self):
        return len(self._results)

