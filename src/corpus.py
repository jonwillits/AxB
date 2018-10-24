import random
from src import config

class CorpusBase(object):
    def __init__(self):

        self.num_tokens = 0

        self.X = []
        self.Y = []

        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_freq_dict = {}

class AxbCorpus(CorpusBase):

    def __init__(self,
                 num_sequences = config.AxB.num_sequences,
                 sample = config.AxB.sample,
                 ab_types = config.AxB.ab_types,
                 x_types = config.AxB.x_types,
                 max_distance = config.AxB.max_distance,
                 min_distance = config.AxB.min_distance,
                 noise = config.AxB.noise,
                 punct = config.AxB.punct,
                 seed = config.AxB.seed,
                 ):

        self.num_tokens = 0

        self.sequence_population = []
        self.sequence_sample = []
        self.X = []
        self.Y = []

        self.vocab_list = []
        self.vocab_index_dict = {}
        self.vocab_freq_dict = {}

        self.num_sequences = num_sequences
        self.sample = sample
        self.ab_types = ab_types
        self.x_types = x_types
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.noise = noise
        self.punct = punct
        self.seed = seed

        self.axb_pair_list = []
        self.a_list = []
        self.b_list = []
        self.x_list = []

        self.stimulus_category_dict = {}
        self.category_item_lists_dict = {'A': [], 'B': [], 'x': [], '.': ['.']}

        self.xsize = self.ab_types*2 + self.x_types + 1
        self.ysize = self.xsize

        self.vocab_list.append('.')
        self.vocab_index_dict['.'] = 0
        self.vocab_freq_dict['.'] = 0

        self.generate_vocab()
        self.generate_sequence_population()
        self.sample_sequences()
        self.count_freqs()

    def generate_vocab(self):

        vocab_counter = 1

        for i in range(self.ab_types):

            a = "A" + str(i+1)
            b = "B" + str(i+1)
            self.axb_pair_list.append((a,b))

            self.a_list.append(a)
            self.vocab_list.append(a)
            self.vocab_index_dict[a] = vocab_counter
            self.vocab_freq_dict[a] = 0
            self.stimulus_category_dict[a] = 'A'
            self.category_item_lists_dict['A'].append(a)
            vocab_counter += 1

            self.b_list.append(b)
            self.vocab_list.append(b)
            self.vocab_index_dict[b] = vocab_counter
            self.vocab_freq_dict[b] = 0
            self.stimulus_category_dict[a] = 'B'
            self.category_item_lists_dict['B'].append(b)
            vocab_counter += 1

        for i in range(self.x_types):
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

        if self.punct:
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
                for j in range(self.x_types):
                    new_sequence = old_sequence.copy()
                    new_sequence.insert(insertion_position,self.x_list[j])
                    new_sequences.append(new_sequence)

        return new_sequences

    def generate_sequence_population(self):

        self.all_sequences = []
        max_distance = self.max_distance
        min_distance = self.min_distance

        for i in range(self.ab_types):
            new_sequence = [self.a_list[i], self.b_list[i]]

            if self.punct:
                new_sequence.append('.')
            self.sequence_population.append(new_sequence)

        current_distance = 0
        replace = True
        while current_distance < min_distance:
            self.sequence_population = self.add_x(self.sequence_population, replace)
            current_distance += 1
            print("not here")
        replace = False
        print(current_distance)
        self.output_sequences()
        while current_distance < max_distance:
            self.sequence_population = self.add_x(self.sequence_population, replace)
            current_distance += 1
            print(current_distance)
            self.output_sequences()


    def sample_sequences(self):

        if self.sample:
            for i in range(self.num_sequences):
                self.sequence_sample.append(random.choice(self.sequence_population))
        else:
            self.sequence_sample = self.sequence_population

    def count_freqs(self):

        for sequence in self.sequence_sample:
            for token in sequence:
                self.vocab_freq_dict[token] += 1

    def output_sequences(self):
        for sequence in self.sequence_population:
            print(sequence)

    def output_freqs(self):
        for item in self.vocab_freq_dict:
            print(item, self.vocab_freq_dict[item])




