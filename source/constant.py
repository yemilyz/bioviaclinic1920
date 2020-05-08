
import os
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',))
_SABDAB_RAW = os.path.join(REPO_DIR, 'sabdab_raw')
SABDAB_SUMMARY_ALL_FILE = os.path.join(_SABDAB_RAW, 'sabdab_summary_all.tsv')
SABDAB_SUMMARY_FILE = os.path.join(_SABDAB_RAW, 'sabdab_summary_filtered.tsv')
SABDAB_DATASET_DIR = os.path.join(_SABDAB_RAW, 'sabdab_dataset')
PDB_DIR = os.path.join(_SABDAB_RAW, 'sabdab_filtered_pdb')
CLEAN_PDB_DIR =  os.path.join(_SABDAB_RAW, 'sabdab_filtered_clean_pdb')
HCHAIN_FASTA_FILE = os.path.join(_SABDAB_RAW, 'sabdab_sequences_VH.fa')
LCHAIN_FASTA_FILE = os.path.join(_SABDAB_RAW, 'sabdab_sequences_VL.fa')
TRAINING_DATA_DIR = os.path.join(REPO_DIR, 'training_data')
DI_LABELS_CSV = os.path.join(TRAINING_DATA_DIR, 'DI_out.csv')
FEATURE_DIR = TRAINING_DATA_DIR
LABELS_DIR = TRAINING_DATA_DIR
MODEL_DIR = os.path.join(REPO_DIR, 'd2v_models')
FIGURE_DIR = os.path.join(REPO_DIR, 'figures')

# Hydropathy index
AA_INDEX_IDS = ['KYTJ820101']

# Feature array from EMBOSS PEPSTATS
FEATURE_ARRAY_NAMES = ['molecularWeight', 'numResidues', 'averageResidueWeight', 
                 'charge', 'isoelectricPoint', 'molExtinctCoeff', 
                 'molExtinctCoeffCystineBridges', 'extinctCoeff', 
                 'extinctCoeffCystineBridges', 'IEIB']

SUPPORTED_GRAPH_TYPES = ['ps', 'hpgl', 'hp7470', 'hp7580', 'meta', 'cps', 
                         'x11', 'tek', 'tekt', 'none', 'data', 'xterm', 'svg']
