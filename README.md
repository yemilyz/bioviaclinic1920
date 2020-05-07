# bioviaclinic1920
### Current Tree View
```bash
├── ml
│   ├── code
│   │   ├── classifiers.py
│   │   ├── constant.py
│   │   ├── datasets.py
│   │   ├── learning_curve.py
│   │   ├── metric_analysis.py
│   │   ├── ml.py
│   │   ├── model_analysis.py
│   │   ├── pareto.py
│   │   ├── plot_results.py
│   │   ├── plotting_params.py
│   |   └── preprocessors.py
│   ├── data
│   |    └── 
│   └── result_10splits
│       └── 
│   └── metric_analysis
│       └── 
├── sabdab_raw
│   ├── sabdab_sequences_VH.fa
│   ├── sabdab_sequences_VL.fa
│   └── sabdab_summary_filtered.tsv
├── source
│   ├── clean_pdb.py
│   ├── constant.py
│   ├── embed.py
│   ├── featurize.py
│   ├── io_fasta.py
│   ├── protparam_features.py
│   ├── sabdab_downloader.py
│   ├── summary_stats.py
│   ├── train_doc2vec_models.py	
│   ├── utils.py
│   └── visualize_embed.py
├── .gitignore
└── README.md
```

The `ml` directory contains the model training and evaluation piplines. The `source` directory contains scripts to download sabdab data, clean sabdab data, and generate physicochemical features and embedding features. The  `sabdab_raw` directory contains all filtered sabdad data, along with their processed counterparts. 