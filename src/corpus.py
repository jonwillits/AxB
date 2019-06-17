import random
from sortedcontainers import SortedSet
from itertools import product

from src import config


# shared between two corpora

def calc_max_position(seqs):
    # must take into consideration any punctuation but not padding
    res = max([len(seq) for seq in seqs]) - 1
    return res


def generate_sequence_sample(corpus):
    if corpus.params.sample_size is not None:
        res = []
        for i in range(corpus.params.sample_size):
            res.append(random.choice(corpus.sequence_population))
        return res
    else:
        return corpus.sequence_population


class MarcusCorpus:
    def __init__(self, params, name):
        self.params = params
        self.name = name
        print('Initializing Marcus corpus with name={}'.format(name))
        #
        self.a_items = ['{}{}'.format(cat, i) for cat, i in product(self.cats[0], range(self.params.ab_cat_size))]
        self.b_items = ['{}{}'.format(cat, i) for cat, i in product(self.cats[1], range(self.params.ab_cat_size))]
        self.items = SortedSet(self.a_items + self.b_items)
        if self.params.punctuation or self.params.punctuation_at_start:
            self.items.add('.')
        #
        self.sequence_population = self.generate_sequence_population()
        self.sequences = generate_sequence_sample(self)  # used for training and eval
        self.num_sequences = len(self.sequences)
        #
        self.positions = list(range(calc_max_position(self.sequences) + 1))

    @property
    def cats(self):
        if self.name == 'train':
            return ['A', 'B']
        elif self.name == 'test':
            return ['C', 'D']
        else:
            raise NotImplementedError('How to assign categories when name is not "train" or "test"?')

    def generate_sequence_population(self):
        res = []
        for i in range(self.params.ab_cat_size):
            for j in range(self.params.ab_cat_size):
                if self.params.pattern == 'abb':
                    sequence = [self.a_items[i], self.b_items[j], self.b_items[j]]
                elif self.params.pattern == 'aab':
                    sequence = [self.a_items[i], self.a_items[i], self.b_items[j]]
                elif self.params.pattern == 'aba':
                    sequence = [self.a_items[i], self.b_items[j], self.a_items[i]]
                else:
                    raise AttributeError('Invalid arg to "pattern".')

                if self.params.punctuation_at_start:
                    sequence.insert(0, ".")
                if self.params.punctuation:
                    sequence.append(".")

                res.append(sequence)
        return res


class AxbCorpus:
    def __init__(self, params, name):
        self.params = params
        self.name = name
        print('Initializing AxB corpus with name={}'.format(name))
        #
        self.cats = ['A', 'x', 'B']
        self.a_items = self.make_items('A')
        self.x_items = self.make_items('x')
        self.b_items = self.make_items('B')
        self.items = SortedSet(self.a_items + self.x_items + self.b_items)
        if self.params.punctuation:
            self.items.add('.')
        #
        self.sequence_population = self.generate_sequence_population()
        self.sequences = generate_sequence_sample(self)  # used for training and eval
        self.num_sequences = len(self.sequences)
        #
        self.positions = list(range(calc_max_position(self.sequences) + 1))

    @property
    def x_cat_size(self):
        if self.name == 'train':
            return self.params.train_x_cat_size
        elif self.name == 'test':
            return self.params.test_x_cat_size
        else:
            raise NotImplementedError('What to do when name is not "train" or "test"?')

    @property
    def max_distance(self):
        if self.name == 'train':
            return self.params.max_distance
        elif self.name == 'test':
            return config.Eval.max_distance
        else:
            raise NotImplementedError('What to do when name is not "train" or "test"?')

    def make_items(self, cat):  # TODO test
        res = []
        if cat == 'A':
            for i in range(self.params.ab_cat_size):
                item = '{}{}'.format(cat, i)
                res.append(item)
        elif cat == 'B':
            for i in range(self.params.ab_cat_size):
                item = '{}{}'.format(cat, i)
                res.append(item)
        elif cat == 'x':
            for i in range(self.x_cat_size):
                item = '{}{}'.format(cat, i)
                res.append(item)
        return res

    def add_x(self, old_sequences, replace):
        longest = 0

        for i in old_sequences:
            if len(i) > longest:
                longest = len(i)

        if self.params.punctuation:
            insertion_position = -2
        else:
            insertion_position = -1

        if replace:
            new_sequences = []
        else:
            new_sequences = old_sequences.copy()

        num_old_sequences = len(old_sequences)

        for i in range(num_old_sequences):

            old_sequence = old_sequences[i]

            if len(old_sequence) == longest:
                for j in range(self.x_cat_size):
                    new_sequence = old_sequence.copy()
                    new_sequence.insert(insertion_position,self.x_items[j])
                    new_sequences.append(new_sequence)

        return new_sequences

    def generate_sequence_population(self):
        res = []

        for i in range(self.params.ab_cat_size):
            new_sequence = [self.a_items[i], self.b_items[i]]
            if self.params.punctuation:
                new_sequence.append('.')
            res.append(new_sequence)

        current_distance = 0
        replace = True
        while current_distance < self.params.min_distance:
            res = self.add_x(res, replace)
            current_distance += 1
        replace = False
        while current_distance < self.max_distance:
            res = self.add_x(res, replace)
            current_distance += 1
        return res