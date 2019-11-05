import csv
import pandas as pd
import constants

# Needs features.csv, sabdab_sequences_VH.fa, and DI_labels.csv

def get_pdb_codes(sequences_file):
    """
    Gets PDB codes from the sequences file and returns them in an array
    """
    codes = []
    for line in sequences_file:
        if line[0] == '>':
            line = line[1:]
            line_split = line.split('_')
            codes.append(line_split[0])
    return codes

def concat(features_file, sequences_file):
    """
    Add sequences to features file into FEATURES_SEQS_CSV
    """
    with open(constants.FEATURES_SEQS_CSV, 'w') as output_file:
        writer = csv.writer(output_file)
        reader = csv.reader(features_file)
        output = []
        row = next(reader)
        row[0] = 'Name'
        output.append(row)
        ind = 0
        codes = get_pdb_codes(sequences_file)
        for row in reader:
            row[0] = codes[ind].upper()
            ind += 1
            output.append(row)
        writer.writerows(output)

def add_DI_labels(features_csv, labels_csv):
    """
    Add DI labels to corresponding sequences
    """
    features = pd.read_csv(features_csv)
    di = pd.read_csv(labels_csv)
    new_df = features.merge(di, on='Name')
    new_file = open(constants.LABELED_DATA_CSV, 'w')
    new_df.to_csv(new_file)

def add_DI_classification_labels(labeled_data):
    """
    Add DI classification (DI < 100 = low, DI ≥ 100 = high)
    """
    labeled_data = open(labeled_data)
    with open(constants.LABELED_DATA_MODIFIED_CSV, 'w') as labeled_data_modified:
        writer = csv.writer(labeled_data_modified)
        reader = csv.reader(labeled_data)
        output = []
        row = next(reader)
        row.append('DI Classification')
        output.append(row)
        for row in reader:
            value = 1
            if (float(row[len(row) - 1]) < 100) :
                value = 0
            row.append(value)
            output.append(row)
        writer.writerows(output)

# features_csv = open(constants.FEATURES_CSV)
# sequences_txt = open(constants.SEQUENCE_FASTA)
# # add sequences to features file
# concat(features_csv, sequences_txt)
# # add DI labels to modified features file
# add_DI_labels(constants.FEATURES_SEQS_CSV, constants.DI_LABELS_CSV)
# # add classification column
# add_DI_classification_labels(constants.LABELLED_DATA_CSV)
# features_csv.close()
# sequences_txt.close()

