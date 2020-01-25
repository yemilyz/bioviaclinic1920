from Bio import PDB
from constant import PDB_DIR, CLEAN_PDB_DIR, SABDAB_SUMMARY_FILE
import glob
import os
import pandas as pd


class ChainSplitter:
    def __init__(self, out_dir=None):
        """ Create parsing and writing objects, specify output directory. """
        self.parser = PDB.PDBParser()
        self.writer = PDB.PDBIO()
        if out_dir is None:
            out_dir = os.path.join(os.getcwd(), "chain_PDBs")
        self.out_dir = out_dir

    def make_pdb(self, pdb_path, pdb_id, chain_letters, overwrite=False, struct=None):
        """ Create a new PDB file containing only the specified chains.

        Returns the path to the created file.

        :param pdb_path: full path to the crystal structure
        :param chain_letters: iterable of chain characters (case insensitive)
        :param overwrite: write over the output file if it exists
        """
        # chain_letters = [chain.upper() for chain in chain_letters]

        # Input/output files
        out_name = "%s_%s.pdb" % (pdb_id, "".join(chain_letters))
        out_path = os.path.join(self.out_dir, out_name)
        # print("OUT PATH:", out_path)
        # plural = "s" if (len(chain_letters) > 1) else ""  # for printing

        # Skip PDB generation if the file already exists
        if (not overwrite) and (os.path.isfile(out_path)):
            # print("Chain%s %s of '%s' already extracted to '%s'." %
            #         (plural, ", ".join(chain_letters), pdb_id, out_name))
            return out_path
        # Get structure, write new file with only given chains
        if struct is None:
            struct = self.parser.get_structure(pdb_id, pdb_path)
        # struct = struct.set_header(pdb_id.upper())
        self.writer.set_structure(struct)
        self.writer.save(out_path, select=SelectChains(chain_letters))

        return out_path

class SelectChains(PDB.Select):
    """ Only accept the specified chains when saving. """
    def __init__(self, chain_letters):
        self.chain_letters = chain_letters

    def accept_chain(self, chain):
        return (chain.get_id() in self.chain_letters)

summary = pd.read_csv(SABDAB_SUMMARY_FILE, sep="\t")
summary = summary.drop_duplicates(subset='pdb', keep='first')

summary.index = summary['pdb'].tolist()
summary = summary[['Hchain',	'Lchain']]
hl_mapping = summary.to_dict('index') 
splitter = ChainSplitter(CLEAN_PDB_DIR)  

for root, _, files in os.walk(PDB_DIR, topdown=False):
    for i, pdb_file in enumerate(files):
        if 'pdb' in pdb_file:
            pdb_path = os.path.join(root, pdb_file)
            pdb_key = pdb_file.split('.')[0]
            chains = list(hl_mapping[pdb_key].values())
            try:
                splitter.make_pdb(pdb_path, pdb_key, chains)
            except:
                print(i, pdb_key)
