import os
import glob
import pandas as pd
import shutil
import errno

from constant import SABDAB_SUMMARY_FILE, \
    REPO_DIR, PDB_DIR, SABDAB_DATASET_DIR

def get_filepaths(filetype='sequence'):
    """
    Traverse the downloaded sabdab dataset and extract the full 
    paths of desired filetypes 
    """
    if filetype =='sequence':
        pattern = "sequence/*.fa"
    elif filetype =='pdb':
        pattern = "structure/*.pdb"
    elif filetype =='chothia':
        pattern = "structure/chothia/*.pdb"
    else:
        pattern = ""
    data = pd.read_csv(SABDAB_SUMMARY_FILE, sep="\t")
    data = data.drop_duplicates(subset='pdb', keep='first')
    files = []
    for root, dirs, _ in os.walk(SABDAB_DATASET_DIR, topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, pattern))
            if filetype =='sequence':
                if files_in_dir:
                    pdb_entries = data.loc[data['pdb'] == name]
                    for _ , pdb_enty in pdb_entries.iterrows():
                        hchain_fa = pdb_enty['Hchain_fa']
                        lchain_fa = pdb_enty['Lchain_fa']
                        h_file = list(filter(lambda x: (hchain_fa in x), files_in_dir))
                        l_file = list(filter(lambda x: (lchain_fa in x), files_in_dir))
                        hl_pair = h_file + l_file
                        if len(hl_pair)==2:
                            files.append(hl_pair)
                        else:
                            print(pdb_enty)
                        # for now, use one sequence per ab 
                        break
            else:
                files += files_in_dir
    return files

def cp_pdb():
    """
    Helper function for moving all .pdb files to one flat directory
    """
    pdb_files = get_filepaths(filetype='pdb')
    print(len(pdb_files))
    for pdb_file in pdb_files:
        pdb_name = pdb_file.split('/')[-1]
        shutil.copyfile(pdb_file, os.path.join(PDB_DIR, pdb_name))
