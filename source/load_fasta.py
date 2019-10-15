from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.sequence import SequenceSet
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.utils.mapping import simplify

from quantiprot.metrics.aaindex import get_aaindex_file, get_aa2charge, get_aa2hydropathy, get_aa2volume
from quantiprot.metrics.basic import identity, average, sum_absolute, uniq_count
from quantiprot.metrics.basic import average
from quantiprot.metrics.basic import identity

import numpy as np
from utils import get_filepaths


def fasta_file_to_seqset(fasta_files, seqtype = "_L|seqres|full"):
    seqset = SequenceSet("my_seqset")
    for fa in fasta_files:
        seqs = load_fasta_file(fa)
        for seq in seqs:
            if seq.identifier.endswith(seqtype):
                seqset.add(seq)
    return seqset

# Prepare Features:
# Build a feature: average polarity (Grantham, 1974), AAindex entry: GRAR740102:
# avg_polarity_feat = Feature(get_aaindex_file("GRAR740102")).then(average)
# sum_abs_charge_feat = Feature(get_aa2charge()).then(sum_absolute)
# avg_hydropathy_feat = Feature(get_aa2hydropathy()).then(average)
# freq_feat = Feature(get_aaindex_file("JOND920101"))
# charge_feat = Feature(get_aa2charge())
# Prepare a FeatureSet
# fs = FeatureSet("simple")
# Add the feature to new feature set:
# fs.add(avg_hydropathy_feat)
# fs.add(sum_abs_charge_feat)
# fs.add(avg_polarity_feat)
# fs.add(freq_feat, name='freequency')
# fs.add(charge_feat, name='charge')

# Process sequences
# fasta_files = get_filepaths(pattern = "*.fa")
# seqset = fasta_file_to_seqset(fasta_files)
# result_seqset = fs(seqset)

# print(np.matrix(columns(result_seqset, feature="hydropathy", transpose=True)))
