import os
import glob

import pandas as pd
from embeddings_reproduction import embedding_tools
from quantiprot.utils.io import load_fasta_file

from constant import HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE, FEATURE_DIR, MODEL_DIR
from io_fasta import get_seq_dict

def embed_sequence(model_path, out_path, k=5, overlap=False):
    embeds_data = []
    for filename in [HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE]:
        seqset = get_seq_dict(filename)
        seqs = list(seqset.values())
        embeds = embedding_tools.get_embeddings(model_path, seqs, k, overlap)
        embeds_data.append(pd.DataFrame.from_records(embeds))
    embeds_pd = pd.concat(embeds_data, axis=1)
    embeds_pd.index = list(seqset.keys())
    embeds_pd['pdb_code'] = list(seqset.keys())
    embeds_pd.to_csv(out_path)

if __name__ == "__main__":
    # model_path = os.path.join(MODEL_DIR, 'original_5_7.pkl')
    # k = int(model_path.split('/')[-1].split('_')[1])
    # print(k)
    # feature_filename = 'feature_embedding_{}.csv'.format(model_path.split('/')[-1].split('.')[0])
    # feature_path = os.path.join(FEATURE_DIR, feature_filename)
    # embed_sequence(model_path, feature_path, k=k)
    for model_path in glob.glob(os.path.join(MODEL_DIR, '*.pkl')):
        print('embedding with', model_path)
        k = int(model_path.split('/')[-1].split('_')[1])
        feature_filename = 'feature_embedding_{}'.format(model_path.split('/')[-1].split('.')[0])
        feature_path = os.path.join(FEATURE_DIR, feature_filename)
        if os.path.exists(feature_path):
            continue
        else:
            try:
                embed_sequence(model_path, feature_path, k=k)
                print('saving features to', feature_path)
            except ValueError:
                print('failed on', feature_filename)
                pass
                
