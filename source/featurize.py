from functools import reduce
from quantiprot.utils.feature import Feature, FeatureSet
from quantiprot.utils.mapping import simplify
from quantiprot.utils.io import load_fasta_file
from quantiprot.metrics.aaindex import get_aaindex_file, get_aa2charge, \
                                       get_aa2hydropathy, get_aa2volume
from quantiprot.metrics.basic import identity, average, sum_absolute, \
                                     uniq_count
from quantiprot.metrics.basic import average
from quantiprot.metrics.basic import identity
from quantiprot.utils.sequence import compact, columns
import numpy as np
import os
import pandas as pd
import ssbio.protein.sequence.properties.residues
import ssbio.utils
import shlex
from Bio import SeqIO
from Bio.Seq import Seq

from utils import get_filepaths
from constant import AA_INDEX_IDS, HCHAIN_FASTA_FILE, LCHAIN_FASTA_FILE, \
                     REPO_DIR, DI_LABELS_CVS, FEATURE_ARRAY_NAMES, \
                     SUPPORTED_GRAPH_TYPES, FEATURE_DIR
from io_fasta import fasta_files_to_seqsets, write_seqset_to_fasta

DATA_DIR = FEATURE_DIR
def build_index_feature_set(aa_index_feats):
    #TODO: add functionality for other feature functions
    # Input: a list of tuples, each containing an amino acid index ID, 
    # the function (see below options), window size (sliding), and default 
    # value for elements beyond the standard 20 aminoacids.
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
    # atom_freq:calculate frequency of occurrencies of a given atomic element 
    # in the data.

    # Prepare a FeatureSet
    fs = FeatureSet("simple")
    for (index, function, window, default) in aa_index_feats:
        feat = Feature(get_aaindex_file(index, default=default)) \
                      .then(eval(function),window=window)
        # Add the feature to the feature set
        fs.add(feat) 
    return fs

def featurize_HLchains(seqset_Hchain, seqset_Lchain, featureset):
    """ Takes in a Sequence Set of Hchains, a Sequence Set of corresponding
        Lchains, and a Feature Set and returns a feature matrix of padded 
        Hchains features concatenated with padded Lchains features.
    """
    result_Hchain = featureset(seqset_Hchain)
    result_Lchain = featureset(seqset_Lchain)

    compact_Hchain = compact(result_Hchain)
    compact_Lchain = compact(result_Lchain)
    for cH, cL in zip(compact_Hchain, compact_Lchain):
        if isinstance(cH.data[0], list):
            cH.data = list(reduce(lambda x,y: x+y, cH.data))
            cL.data = list(reduce(lambda x,y: x+y, cL.data))
        else:
            break
    mat_Hchain = np.matrix(columns(compact_Hchain, transpose=True))
    mat_Lchain = np.matrix(columns(compact_Lchain, transpose=True))

    return pd.DataFrame(mat_Hchain, columns=['VH_'+str(i)+'_HydrophobicMoment' for i in range(mat_Hchain.shape[1])]), \
           pd.DataFrame(mat_Lchain, columns=['VL_'+str(i)+'_HydrophobicMoment' for i in range(mat_Lchain.shape[1])])

def add_DI_labels(di_label_csv, feature_matrix):
    di = pd.read_csv(di_label_csv)
    if (di.columns[0] == 'Name'):
        di.rename(columns={'Name':'pdb_code'}, inplace=True)

    di['pdb_code'] = di['pdb_code'].astype(str).str.lower()
    di['pdb_code'] = di['pdb_code'].map(lambda pdb_code: pdb_code.split('_')[0])
    di = di.drop_duplicates()
    print(di.shape)
    new_df = feature_matrix.merge(di, on='pdb_code')
    return new_df

def classifyDI(row):
    if row['Developability Index (Fv)'] < 99:
        return 0
    else:
        return 1

def add_DI_classification_labels(labeled_matrix):
    labeled_matrix['DI Classification'] = labeled_matrix.apply(lambda row: classifyDI(row), axis=1)
    return labeled_matrix

def get_ids_to_array(seqset_Lchain):
    """ Assumes that the ids of the L chain and H chain are in the same order
    """
    return [name.split('|')[0].split('_')[0] for name in seqset_Lchain.ids()]

def get_sequences_to_array(fname, ftype='fasta', outfile=''):
    file_sequences = open(os.path.join(DATA_DIR, 'sequences_'+outfile+'.'+ftype), "w+")
    for seq_record in SeqIO.parse(fname, "fasta"):
        file_sequences.write(str(seq_record.seq))
        file_sequences.write('\n')
    file_sequences.close()
    return file_sequences

def emboss_pepstats_parse_to_dataframe(infile, chain_type=''):
    """ Get dataframe of pepstats results.
        Args:
            infile: Path to pepstats outfile, string
            chain_type: Variable Light (VL) or Variable Heavy (VH), string 
        Returns:
            df_pepstats: Parsed information from pepstats, dataframe
    """
    # array_ids: List of antibody pdb codes, array array_ids, 
    # read file split on antibodies
    with open(infile) as f:
        antibody_properties = f.read().split('PEPSTATS')[1:]

    info_dict = {}
    for i, data in enumerate(antibody_properties):
        lines = data.split('\n')
        property_dict = {}
        count = 0
        # read antibody properties
        # 2: molecular weight, residues
        # 3: average residue weight, charge
        # 4: isoelectric point
        # 5: A280 Molar Extinction Coefficients
        # 6: A280 Extinction Coefficients 1mg/ml
        # 7: Improbability of expression in inclusion bodies
        for l in lines[2:8]:
            info = l.split('\t')
            clean_info = list(filter(lambda x: x != '', info))
            for property_type in clean_info:
                property_name = chain_type + '_' + FEATURE_ARRAY_NAMES[count]
                property_value_arr = property_type.split('=')[1].split('(reduced)')
                property_value = float(property_value_arr[0])
                count += 1
                property_dict[property_name] = property_value
                if (len(property_value_arr) > 1 \
                    and count < len(FEATURE_ARRAY_NAMES)):
                    property_cys_bridge = property_value_arr[1] \
                                            .split('(cystine bridges)')
                    property_name = \
                        chain_type + '_' + FEATURE_ARRAY_NAMES[count]
                    property_dict[property_name] = float(property_cys_bridge[0])
                    count += 1
        info_dict[i] = property_dict
    df_pepstats = pd.DataFrame.from_dict(info_dict, orient='index')
    return df_pepstats

def emboss_program_FASTA(infile, program, window_size=5, outfile='', \
                        outdir='', outext='', force_rerun=False, \
                        graph=False, graph_type=''):
    """Run EMBOSS program on a FASTA file.
    Args:
        infile: Path to FASTA file, str
        program: Name of program, str
        window_size: Size of window, int
        outfile: Name of output file without extension, str
        outdir: Path to output directory, str
        outext: Extension of results file, str
        force_rerun: Flag to rerun program, boolean
        graph: Flag to make plot, boolean
        graph_type: Plot type, requires graph=True to output graph, str
    Returns:
        str: Path to output file.
    """
    # Create the output file name
    outfile = ssbio.utils.outfile_maker(inname=infile, outname=outfile, outdir=outdir, outext=outext)
    # Run program
    program = '{} {} -outfile="{}"'.format(program, infile, outfile)
    if (graph and graph_type in SUPPORTED_GRAPH_TYPES):
        program = program + ' -graph={} -plot=Yes'.format(graph_type)
    ssbio.utils.command_runner(program, force_rerun_flag=False, outfile_checker='',silent=True)
    return outfile

def emboss_program_parse_df(infile, program, start_index, chain_type=''):
    with open(infile) as f:
        sw_program = f.read().split(program.upper())[1:]
    info_dict = {}
    for index, antibody in enumerate(sw_program):
        lines = antibody.split('\n')
        program_prop_dict = {}
        for i, line in enumerate(lines[start_index:]):
            res_program = line.split('\t')
            clean_res_program = list(filter(lambda x: x != '', res_program))
            program_name = chain_type + '_' + str(i) + '_emboss' + program.capitalize()
            if (len(clean_res_program) > 0 and clean_res_program[0].isdigit()):
                if (len(clean_res_program) == 3):
                    program_prop_dict[program_name] = float(clean_res_program[2]) 
                else: 
                    program_prop_dict[program_name] = float(clean_res_program[1])
        info_dict[index] = program_prop_dict    
    df_program = pd.DataFrame.from_dict(info_dict, orient='index')
    return df_program

def concat_dataframes_and_pdbcodes(array_ids, *args):
    df_list = []
    for arg in args:
        df_list.append(arg)
    df_final = \
        pd.concat(df_list, join='inner', axis=1)
    df_final['pdb_code'] = array_ids
    columns_list = list(df_final.columns)
    columns_list = [columns_list[-1]] + columns_list[:-1]
    df_final = df_final[columns_list]
    return df_final

def main():
    # Testing
    function_list = ["average"]*len(AA_INDEX_IDS)
    windows = [19]*len(AA_INDEX_IDS)
    default = [0]*len(AA_INDEX_IDS)
    aa_index_feats = zip(AA_INDEX_IDS, function_list, windows, default)

    # Process sequences
    seqset_Hchain, seqset_Lchain = load_fasta_file(HCHAIN_FASTA_FILE), load_fasta_file(LCHAIN_FASTA_FILE)

    # Compile ids of antibodies
    array_ids = get_ids_to_array(seqset_Lchain)

    # Compile sequences of antibodies
    array_sequences_Hchain = get_sequences_to_array(HCHAIN_FASTA_FILE, outfile='Hchain')
    array_sequences_Lchain = get_sequences_to_array(LCHAIN_FASTA_FILE, outfile='Lchain')

    # Get features from EMBOSS
    file_pepstats_Hchain = ssbio.protein.sequence.properties.residues.emboss_pepstats_on_fasta(HCHAIN_FASTA_FILE, outfile='pepstats_Hchain', outdir='../data/')
    file_pepstats_Lchain = ssbio.protein.sequence.properties.residues.emboss_pepstats_on_fasta(LCHAIN_FASTA_FILE, outfile='pepstats_Lchain', outdir='../data/')
    
    df_pepstats_Hchain = emboss_pepstats_parse_to_dataframe(file_pepstats_Hchain, 'VH')
    df_pepstats_Lchain = emboss_pepstats_parse_to_dataframe(file_pepstats_Lchain, 'VL')

    file_charge_Hchain = emboss_program_FASTA(HCHAIN_FASTA_FILE, 'charge', outfile='sw_charge_Hchain', outdir='../data/', outext='.charge')
    file_charge_Lchain = emboss_program_FASTA(LCHAIN_FASTA_FILE, 'charge' ,outfile='sw_charge_Lchain', outdir='../data/', outext='.charge')

    df_charge_Hchain = emboss_program_parse_df(file_charge_Hchain, 'charge', 4, chain_type='VH')
    df_charge_Lchain = emboss_program_parse_df(file_charge_Lchain, 'charge', 4, chain_type='VL')

    file_hmoment_Hchain = emboss_program_FASTA(HCHAIN_FASTA_FILE, 'hmoment', outfile='sw_hmoment_Hchain', outdir='../data/', outext='.hmoment')
    file_hmoment_Lchain = emboss_program_FASTA(LCHAIN_FASTA_FILE, 'hmoment', outfile='sw_hmoment_Lchain', outdir='../data/', outext='.hmoment')

    df_hmoment_Hchain = emboss_program_parse_df(file_hmoment_Hchain, 'hmoment', 5, chain_type='VH')
    df_hmoment_Lchain = emboss_program_parse_df(file_hmoment_Lchain, 'hmoment', 5, chain_type='VL')

    print(df_hmoment_Hchain)
    print(df_hmoment_Hchain.shape)
    featureset = build_index_feature_set(aa_index_feats)
    df_aafeatures_Hchain, df_aafeatures_Lchain = \
        featurize_HLchains(seqset_Hchain, seqset_Lchain, featureset)
    feat_mat = concat_dataframes_and_pdbcodes(array_ids, df_pepstats_Hchain, df_pepstats_Lchain, df_charge_Hchain, df_charge_Lchain, df_aafeatures_Hchain, df_aafeatures_Lchain, df_hmoment_Hchain, df_hmoment_Lchain)
    # print(feat_mat)
    print(feat_mat.shape)
    feat_mat.to_csv(os.path.join(DATA_DIR, "features.csv"), index=False)

    labeled_mat = add_DI_labels(DI_LABELS_CVS, feat_mat)
    print('labeled matrix shape', labeled_mat.shape)
    # labeled_mat.to_csv(os.path.join(DATA_DIR, "features_label.csv"), index=False)
    classif_labeled_mat = add_DI_classification_labels(labeled_mat)
    classif_labeled_mat.to_csv(os.path.join(DATA_DIR, "features_label.csv"), index=False)
    print('classified feature matrix shape', classif_labeled_mat.shape)

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
