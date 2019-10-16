"""
Author      : Tom Dougherty
Date        : 2019 October 16
Description : Module to compute sequence features used in ProteinSol.
"""

# Input: fasta file of sequence
# Output: table of features for each sequences

import argparse

from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd

def amino_acid_feats(seq):
    """
    Compute table amino acid compositons.
    Finds composition of all 20 amino acids,
    as well as K-R, D-E, K+R, D+E, K+R-D-E, K+R+D+E, and F+W+Y.

    Input
    ---------------
    seqs -- sequence string

    Output
    ---------------
    feats -- Series of aminoacid compoisiton features
    """
    anal_seq = ProteinAnalysis(seq)
    # Generate counts of amino acids
    aa_counts = anal_seq.count_amino_acids()
    # Add extra features
    aa_counts['K-R'] = aa_counts['K'] - aa_counts['R']
    aa_counts['D-E'] = aa_counts['D'] - aa_counts['E']
    aa_counts['K+R'] = aa_counts['K'] + aa_counts['R']
    aa_counts['D+E'] = aa_counts['D'] + aa_counts['E']
    aa_counts['K+R-D-E'] = aa_counts['K+R'] - aa_counts['D+E']
    aa_counts['K+R+D+E'] = aa_counts['K+R'] + aa_counts['D+E']
    aa_counts['F+W+Y'] = aa_counts['F'] + aa_counts['W'] + aa_counts['Y']
    # Divide by length
    aa_freqs = {k : v / anal_seq.length for k, v in aa_counts.items()}
    # Convert to series
    feats = pd.Series(aa_freqs)
    return feats

# TODO: write functions to compute other ProteinSol features

if __name__ == '__main__':
    # Define command line arguments
    parser = argparse.ArgumentParser(description='Creates table of ProteinSol sequence features')
    parser.add_argument('-s', '--sequences', type=str, help='fasta file of sequences')
    parser.add_argument('-p', '--pyigclassify', type=str, help='txt table from PyIgClassify')
    # Parse arguments
    args = parser.parse_args()
    seq_path = args.sequences
    pic_path = args.pyigclassify
    # Make series of sequences
    if not seq_path is None:
        # Load sequences from fasta file
        seqs = SeqIO.parse(seq_path, "fasta") # Creates a generator
        # Convert to Series of strings
        seqs = pd.Series([str(seq.seq) for seq in seqs])
    elif not pic_path is None:
        # Load table into DataFrame and extract sequence Series
        df = pd.read_csv(pic_path, sep='\t', header=1)
        seqs = df.seqs
    # Generate features
    df = seqs.apply(amino_acid_feats)
    print(df)
    
    # TODO: generate other 8 features
    # length, pI, hydropathy (Kyte and Doolittle, 1982), 
    # absolute charge at pH 7, fold propensity (Uversky et al., 2000), 
    # disorder (Linding et al., 2003), 
    # sequence entropy, and b-strand propensity (Costantini et al., 2006)
    
    # TODO: write df to csv
