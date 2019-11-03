import csv
import pandas as pd

# Needs features.csv, sabdab_sequences_VH.fa, and DI_labels.csv

def get_codes(sequences_file):
    codes = []
    for line in sequences_file:
        if line[0] == '>':
            line = line[1:]
            line_split = line.split('_')
            codes.append(line_split[0])
    return codes

def add_labels(features_csv, labels_csv):
    features = pd.read_csv(features_csv)
    di = pd.read_csv(labels_csv)
    new_df = features.merge(di, on='Name')
    new_file = open('labeled_data.csv', 'w')
    new_df.to_csv(new_file)

def change_labels(labeled_data):
    labeled_data = open(labeled_data)
    with open('labeled_data_modified.csv', 'w') as labeled_data_modified:
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

def concat(features_file, sequences_file):
    with open('features_seq.csv', 'w') as output_file:
        writer = csv.writer(output_file)
        reader = csv.reader(features_file)
        output = []
        row = next(reader)
        row[0] = 'Name'
        output.append(row)
        ind = 0
        codes = get_codes(sequences_file)
        for row in reader:
            row[0] = codes[ind].upper()
            ind += 1
            output.append(row)
        writer.writerows(output)

# features_csv = open('features.csv')
# sequences_txt = open('sabdab_sequences_VH.fa')
# add sequences to features file
# concat(features_csv, sequences_txt)
# add DI labels to modified features file
# add_labels('features_seq.csv', 'DI_labels.csv')
# add classification column
# change_labels('labeled_data.csv')
# features_csv.close()
# sequences_txt.close()

