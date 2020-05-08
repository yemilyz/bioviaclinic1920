import os

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import numpy as np

from constant import FEATURE_DIR, HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE
from io_fasta import get_seq_dict


def generate_features(filename):
    feature_set = {}
    seqs = get_seq_dict(filename)
    colNames = ['aa_percent{}'.format(i) for i in range(20)] + ['aromacity', 'instability',
                'flexibility', 'isoelectric', 'mol_extinct1',
                'mol_extinct2', 'mw', 'gravy', 'ss_faction1', 'ss_faction2',
                'ss_faction3']
    for name, seq in seqs.items():
        analysed_seq = ProteinAnalysis(seq)
        aa_per = analysed_seq.get_amino_acids_percent().values()
        aromacity = analysed_seq.aromaticity()
        instability = analysed_seq.instability_index()
        flexibility = np.average(analysed_seq.flexibility())
        isoelectric = analysed_seq.isoelectric_point()
        mol_extinct1, mol_extinct2 = analysed_seq.molar_extinction_coefficient()
        mw = analysed_seq.molecular_weight()
        gravy = analysed_seq.gravy()
        ss_faction = analysed_seq.secondary_structure_fraction()
        feature = list(aa_per) + [aromacity, instability, flexibility, isoelectric, mol_extinct1, mol_extinct2, mw, gravy] + list(ss_faction)
        feature_set[name] = feature
    feature_set = pd.DataFrame.from_dict(feature_set, orient='index', columns=colNames)
    return feature_set


def main():
    H_features = generate_features(HCHAIN_FASTA_FILE)
    L_features = generate_features(LCHAIN_FASTA_FILE)

    features = H_features.merge(L_features, left_index=True, right_index=True)
    features['name'] = features.index
    features.to_csv(os.path.join(FEATURE_DIR, 'protparam_features.csv'))

if __name__ == "__main__":
    main()