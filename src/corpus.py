import random

from src import config


class MarcusCorpus:
    def __init__(self, params, test):
        self.params = params
        self.item_suffix = 'test' if test else ''
        print('Initializing {} Marcus corpus'.format('test' if test else 'train'))
        #
        self.sequence_population = []
        self.sequence_sample = []
        self.items = []
        self.item2id = {}
        self.num_sequences = None
        self.a_items = []
        self.b_items = []
        self.item2cat = {}
        self.cat2items = {'A': [], 'B': [], '.': ['.']}
        #
        self.generate_vocab()
        self.generate_sequence_population()
        self.generate_sequence_sample()
        #
        if self.params.punctuation or self.params.punctuation_at_start:
            self.items.append('.')
            self.item2id['.'] = len(self.items)
            
    def generate_vocab(self):
        
        vocab_counter = 0

        for i in range(self.params.ab_cat_size):  # B must come first because b_probs are assumed to be first in output
            b = "B" + str(i + 1) + self.item_suffix
            self.b_items.append(b)
            self.items.append(b)
            self.item2id[b] = vocab_counter
            self.item2cat[b] = 'B'
            self.cat2items['B'].append(b)
            vocab_counter += 1

        for i in range(self.params.ab_cat_size):
            a = "A" + str(i + 1) + self.item_suffix
            self.a_items.append(a)
            self.items.append(a)
            self.item2id[a] = vocab_counter
            self.item2cat[a] = 'A'
            self.cat2items['A'].append(a)
            vocab_counter += 1

    def generate_sequence_population(self):

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

                self.sequence_population.append(sequence)

    def generate_sequence_sample(self):
        if self.params.sample_size is not None:
            for i in range(self.params.sample_size):
                self.sequence_sample.append(random.choice(self.sequence_population))
        else:
            self.sequence_sample = self.sequence_population
        self.num_sequences = len(self.sequence_sample)
    

class AxbCorpus:

    def __init__(self, params, test):
        self.params = params
        self.x_cat_size = params.test_x_cat_size if test else params.train_x_cat_size
        self.max_distance = config.Eval.max_distance if test else params.max_distance
        print('Initializing {} AxB corpus with max_distance={}'.format(
            'test' if test else 'train', self.max_distance))
        #
        self.sequence_population = []
        self.sequence_sample = []
        self.items = []
        self.item2id = {}
        self.num_sequences = None
        self.axb_pair_list = []
        self.a_items = []
        self.b_items = []
        self.x_list = []
        self.item2cat = {}
        self.cat2items = {'A': [], 'B': [], 'x': [], '.': ['.']}
        #
        self.generate_vocab()
        self.generate_sequence_population()
        self.generate_sequence_sample()
        #
        if self.params.punctuation:
            self.items.append('.')
            self.item2id['.'] = len(self.items)

    def generate_vocab(self):

        vocab_counter = 0

        for i in range(self.params.ab_cat_size):  # B must come first because b_probs are assumed to be first in output
            b = "B" + str(i + 1)
            self.b_items.append(b)
            self.items.append(b)
            self.item2id[b] = vocab_counter
            self.item2cat[b] = 'B'
            self.cat2items['B'].append(b)
            vocab_counter += 1

        for i in range(self.params.ab_cat_size):
            a = "A" + str(i+1)
            self.a_items.append(a)
            self.items.append(a)
            self.item2id[a] = vocab_counter
            self.item2cat[a] = 'A'
            self.cat2items['A'].append(a)
            vocab_counter += 1

        for i in range(self.params.ab_cat_size):
            a = "A" + str(i + 1)
            b = "B" + str(i + 1)
            self.axb_pair_list.append((a, b))

        for i in range(self.x_cat_size):
            x = "x" + str(i + 1)
            self.x_list.append(x)
            self.items.append(x)
            self.item2id[x] = vocab_counter
            self.item2cat[x] = 'x'
            self.cat2items['x'].append(x)
            vocab_counter += 1

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
                    new_sequence.insert(insertion_position,self.x_list[j])
                    new_sequences.append(new_sequence)

        return new_sequences

    def generate_sequence_population(self):

        for i in range(self.params.ab_cat_size):
            new_sequence = [self.a_items[i], self.b_items[i]]
            if self.params.punctuation:
                new_sequence.append('.')
            self.sequence_population.append(new_sequence)

        current_distance = 0
        replace = True
        while current_distance < self.params.min_distance:
            self.sequence_population = self.add_x(self.sequence_population, replace)
            current_distance += 1
        replace = False
        while current_distance < self.max_distance:
            self.sequence_population = self.add_x(self.sequence_population, replace)
            current_distance += 1

    def generate_sequence_sample(self):
        if self.params.sample_size is not None:
            for i in range(self.params.sample_size):
                self.sequence_sample.append(random.choice(self.sequence_population))
        else:
            self.sequence_sample = self.sequence_population
        self.num_sequences = len(self.sequence_sample)




