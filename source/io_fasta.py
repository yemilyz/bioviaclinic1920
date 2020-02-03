
from quantiprot.utils.io import load_fasta_file
from quantiprot.utils.sequence import SequenceSet, Sequence
from utils import get_filepaths
from constant import HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE, DI_LABELS_CSV
import pandas as pd
from Bio import SeqIO

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
    min_seq = float('inf')
    min_ident = ""
    with open(output_file, 'w') as f:
        for seq in seqset:
            ident = seq.identifier
            data = "".join(seq.data)
            data = data.replace('-', '')
            if len(data) < min_seq:
                min_seq = len(data)
                min_ident = ident
            f.write(">{}\n{}\n".format(ident, data))
    print('min ident', min_ident)
    print('min len', min_seq)


def get_seq_dict(filename):
    seq_dict = {}
    for record in SeqIO.parse(filename, "fasta"):
        seq_dict[record.id[:4]] = str(record.seq)
    return seq_dict

def write_seq_dict_to_posneg_fasta(seq_dict, filename_prefix):
    y = pd.read_csv(DI_LABELS_CSV)
    y.Name = y.Name.str.slice(stop=4)
    y['Developability Index (Fv)'] = (-1)*y['Developability Index (Fv)']
    y['binary_labs'] = y['Developability Index (Fv)'] < y['Developability Index (Fv)'].describe(percentiles=[0.2])[4]
    lab_dict = dict(zip(y.Name.tolist(), y['binary_labs'].tolist()))
    filename_prefix = filename_prefix.split('.')[0]
    pos_filename = "{}_pos.fasta".format(filename_prefix)
    neg_filename = "{}_neg.fasta".format(filename_prefix)
    with open(pos_filename, 'w') as pos, open(neg_filename, 'w') as neg:
        for name, seq in seq_dict.items():
            if lab_dict.get(name):
                print(name)
                pos.write(">{}\n{}\n".format(name, seq))
            else:
                neg.write(">{}\n{}\n".format(name, seq))            


# hd = get_seq_dict(HCHAIN_FASTA_FILE)
# write_seq_dict_to_posneg_fasta(hd, HCHAIN_FASTA_FILE)

if __name__ == "__main__":
    files_paths = get_filepaths()
    print(len(files_paths))
    hseq, lseq = fasta_files_to_seqsets(files_paths)
    write_seqset_to_fasta(hseq, HCHAIN_FASTA_FILE)
    write_seqset_to_fasta(lseq, LCHAIN_FASTA_FILE)
