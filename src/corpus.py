import random


class AxbCorpus:

    def __init__(self, params, num_x_types):
        self.params = params
        self.num_x_types = num_x_types

        self.num_tokens = 0
        self.sequence_population = []
        self.sequence_sample = []
        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_freq_dict = {}
        self.num_sequences = None
        self.axb_pair_list = []
        self.a_list = []
        self.b_list = []
        self.x_list = []
        self.stimulus_category_dict = {}
        self.category_item_lists_dict = {'A': [], 'B': [], 'x': [], '.': ['.']}
        if self.params.punct:
            self.vocab_list.append('.')
            self.vocab_index_dict['.'] = 0
            self.vocab_freq_dict['.'] = 0
        #
        self.generate_vocab()
        self.generate_sequence_population()
        self.generate_sequence_sample()

    def generate_vocab(self):

        vocab_counter = 1

        for i in range(self.params.num_ab_types):
            a = "A" + str(i+1)
            self.a_list.append(a)
            self.vocab_list.append(a)
            self.vocab_index_dict[a] = vocab_counter
            self.vocab_freq_dict[a] = 0
            self.stimulus_category_dict[a] = 'A'
            self.category_item_lists_dict['A'].append(a)
            vocab_counter += 1

        for i in range(self.params.num_ab_types):
            b = "B" + str(i + 1)
            self.b_list.append(b)
            self.vocab_list.append(b)
            self.vocab_index_dict[b] = vocab_counter
            self.vocab_freq_dict[b] = 0
            self.stimulus_category_dict[b] = 'B'
            self.category_item_lists_dict['B'].append(b)
            vocab_counter += 1

        for i in range(self.params.num_ab_types):
            a = "A" + str(i + 1)
            b = "B" + str(i + 1)
            self.axb_pair_list.append((a, b))

        for i in range(self.num_x_types):
            x = "x" + str(i + 1)
            self.x_list.append(x)
            self.vocab_list.append(x)
            self.vocab_index_dict[x] = vocab_counter
            self.stimulus_category_dict[x] = 'x'
            self.category_item_lists_dict['x'].append(x)
            self.vocab_freq_dict[x] = 0
            vocab_counter += 1

    def add_x(self, old_sequences, replace):
        longest = 0

        for i in old_sequences:
            if len(i) > longest:
                longest = len(i)

        if self.params.punct:
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
        max_distance = self.params.max_distance
        min_distance = self.params.min_distance

        for i in range(self.params.num_ab_types):
            new_sequence = [self.a_list[i], self.b_list[i]]

            if self.params.punct:
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




