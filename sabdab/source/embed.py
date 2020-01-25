from embeddings_reproduction import embedding_tools
from quantiprot.utils.io import load_fasta_file
from constant import HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE
import pandas as pd

embeds_data = []
for filename in [HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE]:
    seqset = load_fasta_file(HCHAIN_FASTA_FILE)
    seqs = seqset.columns(default='', transpose=True)
    seqs = [''.join(s) for s in seqs]
    embeds = embedding_tools.get_embeddings('doc2vec_models/original_5_7.pkl', seqs, k=5, overlap=False)
    embeds_data.append(pd.DataFrame.from_records(embeds))

ids = [sid.split('_')[0] for sid in seqset.ids()]
embeds_pd = pd.concat(embeds_data, axis=1)
embeds_pd.to_csv('../features/embed_5_7.csv')