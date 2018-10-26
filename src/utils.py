def output_sequences(sequence_list):
    for sequence in sequence_list:
        print(sequence)

def count_freqs(sequence_list):

    for sequence in sequence_list:
        for token in sequence:
            self.vocab_freq_dict[token] += 1

def output_freqs(freq_dict):
    for item in freq_dict:
        print(item, freq_dict[item])
