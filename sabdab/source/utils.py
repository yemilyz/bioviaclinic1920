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
    else:
        pattern = ""
    data = pd.read_csv(SABDAB_SUMMARY_FILE, sep="\t")
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
                        hl_pair = sorted(list(filter(lambda x: ((hchain_fa in x) or (lchain_fa in x)), files_in_dir)))
                        files.append(hl_pair)
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
    pdb_files_split = [pdb_files[i:i + 100] for i in range(0, len(pdb_files), 100)]
    for i, pdb_files in enumerate(pdb_files_split):
        pdb_batch_dir = os.path.join(PDB_DIR, "batch"+str(i).zfill(2))
        try:
            os.mkdir(pdb_batch_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            pass
        for pdb_file in pdb_files:
            pdb_name = pdb_file.split('/')[-1]
            shutil.copyfile(pdb_file, os.path.join(pdb_batch_dir, pdb_name))
