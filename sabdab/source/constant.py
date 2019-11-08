
import os
# H ARGP820101
# D Hydrophobicity index (Argos et al., 1982)
# R LIT:0901079b PMID:7151796

# H BIGC670101
# D Residue volume (Bigelow, 1967)

# H CHAM820101
# D Polarizability parameter (Charton-Charton, 1982)

# H CHAM820102
# D Free energy of solution in water, kcal/mole (Charton-Charton, 1982)

# H CHAM830107
# D A parameter of charge transfer capability (Charton-Charton, 1983)
# R LIT:0907093b PMID:6876837

# H FASG760101
# D Molecular weight (Fasman, 1976)

# H FASG760104
# D pK-N (Fasman, 1976)

# H FASG760105
# D pK-C (Fasman, 1976)

# H FAUJ880108
# D Localized electrical effect (Fauchere et al., 1988)
# R LIT:1414114 PMID:3209351

# H FAUJ880111
# D Positive charge (Fauchere et al., 1988)
# R LIT:1414114 PMID:3209351

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',))
SABDAB_SUMMARY_ALL_FILE = os.path.join(REPO_DIR, 'sabdab_summary_all.tsv')
SABDAB_SUMMARY_FILE = os.path.join(REPO_DIR, 'sabdab_summary_filtered.tsv')
SABDAB_DATASET_DIR = os.path.join(REPO_DIR, 'sabdab_filtered_dataset')
PDB_DIR = os.path.join(REPO_DIR, 'sabdab_filtered_pdb')


HCHAIN_FASTA_FILE = os.path.join(REPO_DIR, 'sabdab_sequences_VH.fa')
LCHAIN_FASTA_FILE = os.path.join(REPO_DIR, 'sabdab_sequences_VL.fa')

# Hydrophobicity index, Residue Volume, Polarizability Parameter
# Free Energy of Solution in Water (kCal/mole), Charge Transfer Capability Parameter,
# Molecular Weight, pK-N, pK-C, Localized Electrical Effect, Positive Charge
AA_INDEX_IDS = ['ARGP820101', 'BIGC670101', 'CHAM820101', 
        'CHAM820102', 'CHAM830107', 'FASG760101',
        'FASG760104', 'FASG760105',  'FAUJ880108', 'FAUJ880111', 'FAUJ880112']
