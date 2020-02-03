import multiprocessing
import os
import pickle
import random

import gensim
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from embeddings_reproduction import embedding_tools

from quantiprot.utils.io import load_fasta_file
from constant import HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE


def train(X, k, window):
    name_list = [X, str(k), str(window)]
    if os.path.isfile('../outputs/docvec_models/' + '_'.join(name_list) + '.pkl'):
        return
    print('X\t\tk\twindow')
    print(name_list[0] + '\t\t' + '\t'.join(name_list[1:]))
    kmer_hypers = {'k':k, 
                   'overlap':False,
                   'merge':False}
    model_hypers = {'size': 64,
                    'min_count': 0,
                    'iter': 25,
                    'window':window,
                    'workers': 4}
    documents = embedding_tools.Corpus(sequence_dict[X], kmer_hypers)
    model = Doc2Vec(**model_hypers)
    model.build_vocab(documents)
    model.train(documents)
    model.save('doc2vec_models/' + '_'.join(name_list) + '.pkl')


seqset = load_fasta_file(HCHAIN_FASTA_FILE)
seqs = seqset.columns(default='', transpose=True)
seq_data = pd.DataFrame([''.join(s) for s in seqs], columns=['sequence'])

sequence_dict = {'original': seq_data}


for X in sequence_dict.keys():
    for k in range(1, 6):
        for window in range(4, 16, 2):
            train(X, k, window)