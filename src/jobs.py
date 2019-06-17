
from src.evaluation import update_cat_and_item_pps
from src.results import Results
from src import config


def train_loop(rnn, master_vocab, pretrain_seqs=()):
    corpus2results = {corpus.name: Results(corpus) for corpus in master_vocab.corpora}

    # calc seqs_pp + item_pp + cat_pp before training
    seqs_pp = rnn.train_epoch(master_vocab.train_seqs, train=False)  # evaluate seqs_pp before training
    update_cat_and_item_pps(rnn, master_vocab, corpus2results)

    # pretrain
    if len(pretrain_seqs) > 0:
        rnn.train_epoch(pretrain_seqs, train=True)  # pretrain after adding first populating corpus2results

    # train + eval loop
    for epoch in range(rnn.params.num_epochs):

        # train
        if config.Verbosity.seqs_pp:
            print('seqs_pp={}'.format(seqs_pp))
        seqs_pp = rnn.train_epoch(master_vocab.train_seqs, train=True)

        # eval
        update_cat_and_item_pps(rnn, master_vocab, corpus2results)

    return corpus2results
