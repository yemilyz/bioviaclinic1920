from functools import reduce
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.sequence import SequenceSet, Sequence
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.utils.mapping import simplify
from quantiprot.metrics.aaindex import get_aaindex_file, get_aa2charge, get_aa2hydropathy, get_aa2volume
from quantiprot.metrics.basic import identity, average, sum_absolute, uniq_count
from quantiprot.metrics.basic import average
from quantiprot.metrics.basic import identity
from quantiprot.utils.sequence import compact, columns

import numpy as np
from utils import get_filepaths
from constant import AA_INDEX_IDS


def fasta_file_to_sequence(fasta_file, seqtype):
    """
    Takes a list of 2 fasta file paths and loads the files. Then
    takes the matching sequence type (region or full) and adds the 2 sequences
    into a seqset
    """
    seqs = load_fasta_file(fasta_file)
    for seq in seqs:
        if seqtype in seq.identifier:
            return seq

# def concat_HL_pairs(seqs):
#     """
#     Takes a SequenceSet object of size 2 and makes a new Sequence object by
#     concatenating the 2 given sequences
#     """
#     identifier = seqs[0].identifier.split('|')[0].split('_')[0]
#     # identifier = seqs[0].identifier.split('|')[0] + seqs[1].identifier.split('|')[0].split('_')[-1]
#     feature = seqs[0].feature
#     data = seqs[0].data + seqs[1].data
#     return Sequence(identifier, feature, data)

def fasta_files_to_seqsets(fasta_files, seqtype = 'seqres|region:'):
    """ Wrapper function to take in a list of lists of fasta filepath pairs and
    makes a SequenceSet containing all antibody heavy+light chain sequences
    """
    seqset_Hchain = SequenceSet("Hchain Sequences")
    seqset_Lchain = SequenceSet("Lchain Sequences")
    for fasta_files_Hchain, fasta_files_Lchain in fasta_files:
        sequence_Hchain = fasta_file_to_sequence(fasta_files_Hchain, seqtype)
        sequence_Lchain = fasta_file_to_sequence(fasta_files_Lchain, seqtype)
        seqset_Hchain.add(sequence_Hchain)
        seqset_Lchain.add(sequence_Lchain)
    return seqset_Hchain, seqset_Lchain

def build_index_feature_set(aa_index_feats):
    #TODO: add functionality for other feature functions
    # Input: a list of tuples, each containing an amino acid index ID, 
    # the function (see below options), window size, and default value
    # for elements beyond the standard 20 aminoacids.
    # Output: a FeatureSet

    # Function options: 
    # identity: return the data itself.
    # absolute: calculate the absolute values of the data.
    # sum_absolute: calculate the sum of absolute values of the data.
    # average: calculate the arithmetic average of the data.
    # average_absolute: calculate the average of absolute values of the data.
    # uniq_count: count number of unique elements in the data.
    # uniq_average: calculate number of unique elements per length in the data.
    # atom_count: count occurrencies of a given atomic element in the data.
    # atom_freq:calculate frequency of occurrencies of a given atomic element in the data.

    # Prepare a FeatureSet
    fs = FeatureSet("simple")
    for (index,function,window, default) in aa_index_feats:
        feat = Feature(get_aaindex_file(index, default=default)).then(eval(function),window=window)
        # Add the feature to the feature set
        fs.add(feat) 
    return fs

def featurize_HLchains(seqset_Hchain, seqset_Lchain, featureset):
    """ Takes in a Sequence Set of Hchains, a Sequence Set of corresponding
    Lchains, and a Feature Set and returns a feature matrix of padded Hchains features
    concatenated with padded Lchains features.
    """
    result_Hchain = featureset(seqset_Hchain)
    result_Lchain = featureset(seqset_Lchain)
    compact_Hchain = compact(result_Hchain)
    compact_Lchain = compact(result_Lchain)
    for cH, cL in zip(compact_Hchain, compact_Lchain):
        cH.data = list(reduce(lambda x,y: x+y, cH.data))
        cL.data = list(reduce(lambda x,y: x+y, cL.data))
    mat_Hchain = np.matrix(columns(compact_Hchain, transpose=True))
    mat_Lchain = np.matrix(columns(compact_Lchain, transpose=True))
    feat_mat = np.concatenate((mat_Hchain, mat_Lchain), axis=1)
    return feat_mat

def main():
    # testing
    function_list = ["average"]*len(AA_INDEX_IDS)
    windows = [9]*len(AA_INDEX_IDS)
    default = [0]*len(AA_INDEX_IDS)
    aa_index_feats = zip(AA_INDEX_IDS, function_list, windows, default)
    # Process sequences
    fasta_files = get_filepaths(filetype='sequence')
    seqset_Hchain, seqset_Lchain = fasta_files_to_seqsets(fasta_files)
    featureset = build_index_feature_set(aa_index_feats)
    feat_mat = featurize_HLchains(seqset_Hchain, seqset_Lchain, featureset)
    print('final feature matrix shape', feat_mat.shape)


if __name__ == '__main__':
    main()

# Prepare Features:
# Build a feature: average polarity (Grantham, 1974), AAindex entry: GRAR740102:
# avg_polarity_feat = Feature(get_aaindex_file("GRAR740102")).then(average)
# sum_abs_charge_feat = Feature(get_aa2charge()).then(sum_absolute)
# avg_hydropathy_feat = Feature(get_aa2hydropathy()).then(average)
# freq_feat = Feature(get_aaindex_file("JOND920101"))
# charge_feat = Feature(get_aa2charge())
# # Prepare a FeatureSet
# fs = FeatureSet("simple")
# # Add the feature to new feature set:
# fs.add(avg_hydropathy_feat)
# fs.add(sum_abs_charge_feat)
# fs.add(avg_polarity_feat)
# fs.add(freq_feat, name='freequency')
# fs.add(charge_feat, name='charge')