
from src.evaluation import update_cat_and_item_pps
from src.results import Results
from src import config


def train_loop(srn, master_vocab):
    corpus2results = {corpus.name: Results(corpus) for corpus in master_vocab.corpora}

    # calc seqs_pp + item_pp + cat_pp before training
    seqs_pp = srn.train_epoch(master_vocab.train_seqs, train=False)  # evaluate seqs_pp before training
    update_cat_and_item_pps(srn, master_vocab, corpus2results)

    # train + eval loop
    for epoch in range(srn.params.num_epochs):

        # train
        if config.Verbosity.seqs_pp:
            print('seqs_pp={}'.format(seqs_pp))
        seqs_pp = srn.train_epoch(master_vocab.train_seqs, train=True)

        # eval
        update_cat_and_item_pps(srn, master_vocab, corpus2results)

    return corpus2results
