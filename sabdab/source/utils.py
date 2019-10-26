import os
import glob
import pandas as pd
from constant import SABDAB_SUMMARY_FILE, REPO_DIR

def get_filepaths(filetype='sequence'):
    if filetype =='sequence':
        pattern = "sequence/*.fa"
    elif filetype =='pdb':
        pattern = "structure/*.pdb"
    else:
        pattern = ""
    data = pd.read_csv(SABDAB_SUMMARY_FILE, sep="\t")
    repo_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',))
    files = []
    for root, dirs, _ in os.walk(os.path.join(repo_dir, 'sabdab_dataset'), topdown=False):
        for name in dirs:
            files_in_dir = glob.glob(os.path.join(root, name, pattern))
            if filetype =='sequence':
                if files_in_dir:
                    pdb_entries = data.loc[data['pdb'] == name]
                    for _ , pdb_enty in pdb_entries.iterrows():
                        hchain_fa = pdb_enty['Hchain_fa']
                        lchain_fa = pdb_enty['Lchain_fa']
                        hl_pair = sorted(list(filter(lambda x: ((hchain_fa in x) or (lchain_fa in x)), files_in_dir)))
                        files.append(hl_pair)
                else:
                    files += files_in_dir
    return files

