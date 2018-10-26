from src import corpus
#from src import rnn
from src import utils

verbose = True

axb_corp = corpus.AxbCorpus()
utils.output_sequences(axb_corp.sequence_sample)
utils.output_sequences(axb_corp.index_sequence_list)

#srn = rnn.RNN(axb_corp)
#srn.train(verbose)

