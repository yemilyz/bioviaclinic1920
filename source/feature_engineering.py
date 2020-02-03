import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from constant import DI_LABELS_CSV, DATA_DIR, HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE
from io_fasta import get_seq_dict


if __name__ == "__main__":
    d = get_seq_dict(HCHAIN_FASTA_FILE)

    y = pd.read_csv(DI_LABELS_CSV)
    y.Name = y.Name.str.slice(stop=4)
    y['Developability Index (Fv)'] = (-1)*y['Developability Index (Fv)']
    y['binary_labs'] = y['Developability Index (Fv)'] >= y['Developability Index (Fv)'].describe(percentiles=[0.8])[5]

    feature = pd.DataFrame({'Name': list(d.keys()), 'length': [len(v) for v in d.values()]})
    feature.index = feature['Name']
    feature = feature.loc[y.Name]
    feature['labels'] = y['Developability Index (Fv)'].tolist()
    feature['binary_labs'] = y['binary_labs'].tolist()

    cmap='winter'

    feature.plot(kind='scatter', x='length', y='labels', c=feature.labels,  cmap=cmap)
    plt.show()


