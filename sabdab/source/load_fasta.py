from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.sequence import SequenceSet, Sequence
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.utils.mapping import simplify

from quantiprot.metrics.aaindex import get_aaindex_file, get_aa2charge, get_aa2hydropathy, get_aa2volume
from quantiprot.metrics.basic import identity, average, sum_absolute, uniq_count
from quantiprot.metrics.basic import average
from quantiprot.metrics.basic import identity

import numpy as np
from utils import get_filepaths


def fasta_files_to_sequence(fasta_files_pair, seqtype):
    """
    Takes a list of 2 fasta file paths and loads the files. Then
    takes the matching sequence type (region or full) and adds the 2 sequences
    into a seqset
    """
    seqset = SequenceSet("HL_pair")
    for fa in fasta_files_pair:
        seqs = load_fasta_file(fa)
        for seq in seqs:
            if seqtype in seq.identifier:
                seqset.add(seq)
                break
    return concat_HL_pairs(seqset)

def concat_HL_pairs(seqs):
    """
    Takes a SequenceSet object of size 2 and makes a new Sequence object by
    concatenating the 2 given sequences
    """
    identifier = seqs[0].identifier.split('|')[0].split('_')[0]
    # identifier = seqs[0].identifier.split('|')[0] + seqs[1].identifier.split('|')[0].split('_')[-1]
    feature = seqs[0].feature
    data = seqs[0].data + seqs[1].data
    return Sequence(identifier, feature, data)
    
def fasta_files_to_seqset(fasta_files, seqtype = 'seqres|region:'):
    """ Wrapper function to take in a list of lists of fasta filepath pairs and
    makes a SequenceSet containing all antibody heavy+light chain sequences
    """
    seqset = SequenceSet("sabdab_seqset")
    for fasta_files_pair in fasta_files:
        sequence = fasta_files_to_sequence(fasta_files_pair, seqtype)
        seqset.add(sequence)
    return seqset

# Process sequences
fasta_files = get_filepaths(filetype='sequence')
print(len(fasta_files))
seqset = fasta_files_to_seqset(fasta_files)







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


# result_seqset = fs(seqset)

# print(np.matrix(columns(result_seqset, feature="hydropathy", transpose=True)))
