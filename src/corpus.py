import random

from src import config


class AxbCorpus:

    def __init__(self, params, test):
        self.params = params
        self.num_x_types = params.num_x_test_types if test else params.num_x_train_types
        self.max_distance = config.Eval.max_distance if test else params.max_distance
        print('Initializing {} corpus with max_distance={}'.format(
            'test' if test else 'train', self.max_distance))

        self.sequence_population = []
        self.sequence_sample = []
        self.types = []
        self.type2id = {}
        self.num_sequences = None
        self.axb_pair_list = []
        self.a_list = []
        self.b_list = []
        self.x_list = []
        self.type2cat = {}
        self.cat2types = {'A': [], 'B': [], 'x': [], '.': ['.']}
        #
        self.generate_vocab()
        self.generate_sequence_population()
        self.generate_sequence_sample()
        #
        if self.params.punctuation:
            self.types.append('.')
            self.type2id['.'] = len(self.types)

    def generate_vocab(self):

        vocab_counter = 0

        for i in range(self.params.num_ab_types):  # B must come first because b_probs are assumed to be first in output
            b = "B" + str(i + 1)
            self.b_list.append(b)
            self.types.append(b)
            self.type2id[b] = vocab_counter
            self.type2cat[b] = 'B'
            self.cat2types['B'].append(b)
            vocab_counter += 1

        for i in range(self.params.num_ab_types):
            a = "A" + str(i+1)
            self.a_list.append(a)
            self.types.append(a)
            self.type2id[a] = vocab_counter
            self.type2cat[a] = 'A'
            self.cat2types['A'].append(a)
            vocab_counter += 1

        for i in range(self.params.num_ab_types):
            a = "A" + str(i + 1)
            b = "B" + str(i + 1)
            self.axb_pair_list.append((a, b))

        for i in range(self.num_x_types):
            x = "x" + str(i + 1)
            self.x_list.append(x)
            self.types.append(x)
            self.type2id[x] = vocab_counter
            self.type2cat[x] = 'x'
            self.cat2types['x'].append(x)
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
                for j in range(self.num_x_types):
                    new_sequence = old_sequence.copy()
                    new_sequence.insert(insertion_position,self.x_list[j])
                    new_sequences.append(new_sequence)

        return new_sequences

    def generate_sequence_population(self):
        max_distance = self.max_distance
        min_distance = self.params.min_distance

        for i in range(self.params.num_ab_types):
            new_sequence = [self.a_list[i], self.b_list[i]]

            if self.params.punctuation:
                new_sequence.append('.')
            self.sequence_population.append(new_sequence)

        current_distance = 0
        replace = True
        while current_distance < min_distance:
            self.sequence_population = self.add_x(self.sequence_population, replace)
            current_distance += 1
        replace = False
        while current_distance < max_distance:
            self.sequence_population = self.add_x(self.sequence_population, replace)
            current_distance += 1

    def generate_sequence_sample(self):
        if self.params.sample_size is not None:
            for i in range(self.params.sample_size):
                self.sequence_sample.append(random.choice(self.sequence_population))
        else:
            self.sequence_sample = self.sequence_population
        self.num_sequences = len(self.sequence_sample)




