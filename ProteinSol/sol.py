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

def KD_hydropathy(seq):
    """
    Compute the Kyte-Doolittle hydropathy index (JMB 1982) 
    normalized according to Uversky (Proteins 2000).
    Code adapted from https://github.com/maxhebditch/abpred.
    
    Input
    ---------------
    seqs -- sequence string

    Output
    ---------------
    avg_KD -- Normalized KD hydropathy index averaged across residues
    """
    # KD values taken from ABPred code (based on Jain et al 2017)
    aa_KD = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E':-3.5,'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9,
    'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7,
    'V': 4.2, 'W': -0.9, 'Y': -1.3, 'X': 0.0}

    # Normalize the KD hydropathy values to be in between 0 and 1 
    for key in aa_KD.keys():
        aa_KD[key]+=4.5
        aa_KD[key]/=9.0
    
    # Generate counts of amino acids
    anal_seq = ProteinAnalysis(seq)
    aa_counts = anal_seq.count_amino_acids()

    # Sum KD hydropathy values for the entire sequence
    avg_KD = 0.0
    for key in aa_counts.keys():
        if key in aa_KD.keys():
            avg_KD += aa_KD[key]*aa_counts[key]

    return avg_KD
    

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
