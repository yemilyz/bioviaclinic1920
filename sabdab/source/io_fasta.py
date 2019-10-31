
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.sequence import SequenceSet, Sequence

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


def write_seqset_to_fasta(seqset, output_file):
    with open(output_file, 'w') as f:
        for seq in seqset:
            ident = seq.identifier
            data = "".join(seq.data)
            f.write(">{}\n{}\n".format(ident, data))


    