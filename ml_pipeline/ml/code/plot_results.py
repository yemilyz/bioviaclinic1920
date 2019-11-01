"""
Author      : Yi-Chieh Wu
Class       : HMC CS 121
Date        : 2018 Sep 20
Description : ML Comparison Plots
"""

# python modules
import os
import argparse
import json

# pandas and pyplot modules
import pandas as pd
import matplotlib.pyplot as plt

# local ML modules
import datasets
import preprocessors
import classifiers

######################################################################
# functions
######################################################################

def get_parser():
    """Make argument parser."""

    parser = argparse.ArgumentParser()

    # positional arguments
    parser.add_argument("dataset",
                        metavar="<dataset>",
                        choices=datasets.DATASETS,
                        help="[{}]".format(' | '.join(datasets.DATASETS)))

    # optional arguments
    if preprocessors.PREPROCESSORS:
        parser.add_argument("-p", "--preprocessor", dest="preprocessors",
                            metavar="<preprocessor>", 
                            default=[], action="append",
                            choices=preprocessors.PREPROCESSORS,
                            help="[{}]".format(' | '.join(preprocessors.PREPROCESSORS)))
    else:
        parser.set_defaults(preprocessors=[])

    return parser


def load_json(dataset, preprocessor_list):
    """Load results from json file."""
    results = {}
    for classifier in classifiers.CLASSIFIERS:
        prefix = os.path.join("results", '_'.join([dataset] + preprocessor_list + [classifier]))
        json_file = prefix + "_results.json"
        with open(json_file, 'r') as infile:
            res = json.load(infile)
            results[classifier] = res
    return results


def plot(train_df, test_df, dataset, preprocessor_list):
    """Plot training and test performance."""
    fig, (ax1, ax2) = plt.subplots(2,1)

    # bar graphs
    train_df.plot.bar(rot=0, legend=False, ax=ax1)
    #ax1.legend(train.keys(), bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax1.set_ylabel("score")
    ax1.set_ylim([0,1])
    ax1.set_title("Training Performance")

    test_df.plot.bar(rot=0, legend=False, ax=ax2)
    #ax2.legend(test.keys(), bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
    ax2.set_ylabel("score")
    ax2.set_ylim([0,1])
    ax2.set_title("Test Performance")

    # legend
    handles, labels = ax2.get_legend_handles_labels() # objects in last axis
    fig.legend(handles, labels,
               loc="center right", borderaxespad=0,
               title="Classifiers")

    # title and layout
    plt.suptitle(dataset, fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.85)    # make space for title
    plt.subplots_adjust(right=0.8)   # make space for legend

    # save and show
    prefix = os.path.join("results", '_'.join([dataset] + preprocessor_list))
    figname = prefix + "_results.png"
    plt.savefig(figname)
    plt.show()


######################################################################
# main
######################################################################

def main():
    # parse arguments
    parser = get_parser()
    args = parser.parse_args()

    # load json
    results = load_json(args.dataset, args.preprocessors)

    # make dataframe
    index = ["accuracy", "precision", "recall", "f1"]
    train = {}
    test = {}
    for classifier, res in results.items():
        train[classifier] = res["scores_train"]
        test[classifier] = res["scores_test"]
    train_df = pd.DataFrame(train, index=index)
    test_df = pd.DataFrame(test, index=index)

    # make plot
    plot(train_df, test_df, args.dataset, args.preprocessors)
    
if __name__ == "__main__":
    main()
