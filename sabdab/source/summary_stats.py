import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import argparse
import errno

from constant import SABDAB_SUMMARY_FILE, REPO_DIR, SABDAB_SUMMARY_ALL_FILE

def plot_all_histograms(data, figdir):
    """
    Plots and saves all relevant histograms of sabdab summary file
    """
    categorical_cols = ['HLchain', 'Hchain', 'Lchain', 'model', 'heavy_species',
        'light_species','method','engineered', 'light_ctype', 'antigen_type']
    numeric_cols = ['date', 'resolution', 'r_free', 'r_factor']
    for category in categorical_cols:
        value_counts = data[category].value_counts(dropna=True)
        if value_counts.size > 30:
            value_counts = value_counts.iloc[:30]
        fig = plt.figure()
        value_counts.plot(title = category, kind='bar')
        fig.tight_layout()
        plt.savefig(os.path.join(figdir, '{}_hist.png'.format(category)))
        plt.close()
    for numcol in numeric_cols:
        if numcol != 'date':
            data[numcol] = pd.to_numeric(data[numcol], errors ='coerce')
        n_bins = 20
        data[numcol].hist(bins=n_bins)
        plt.title(numcol)
        plt.savefig(os.path.join(figdir, '{}_hist.png'.format(numcol)))
        plt.close()

if __name__ == '__main__':
    figdir = os.path.join(REPO_DIR, 'figures')
    try:
        os.mkdir(figdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        pass
    filepath = SABDAB_SUMMARY_ALL_FILE
    dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%y')
    data = pd.read_csv(filepath, sep="\t", parse_dates=['date'], date_parser=dateparse)
    data_filtered = data.dropna(subset = ['Hchain', 'Lchain'])
    data_filtered = data_filtered.loc[data_filtered['resolution']!='NOT']
    data_filtered['HLchain'] = data_filtered['Hchain'] + data_filtered['Lchain']
    data_filtered_res = data_filtered.loc[pd.to_numeric(data_filtered['resolution'], errors ='coerce')< 3]
    plot_all_histograms(data_filtered_res, figdir)

    data_filtered_res['Hchain_fa'] = data_filtered_res['pdb'] + '_' + data_filtered_res['Hchain'] + "_VH.fa"
    data_filtered_res['Lchain_fa'] = data_filtered_res['pdb'] + '_' + data_filtered_res['Lchain'] + "_VL.fa"

    data_filtered_res.to_csv(SABDAB_SUMMARY_FILE, sep='\t', index=False)


