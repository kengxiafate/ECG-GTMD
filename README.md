
# ECG-GTMD
This repository provides the source code ECG-GTMD
# Introduction
- **./data_preprocessing**：Carry out the preprocessing work of the dataset.

- **./data_provider**:Create and load experimental data.

- **./layers**:The reference implementations of each component of model.

- **run.py**:Add parameters and start the experiment.
# Datasets 
All datasets have been made public on physionet.

- PTB dataset:https://physionet.org/content/ptbdb/1.0.0/

- PTB-XLdataset:https://physionet.org/content/ptb-xl/1.0.3/

- MIT-BIH:https://physionet.org/content/mitdb/1.0.0/

- Chapman:https://physionet.org/content/ecg-arrhythmia/1.0.0/
# Getting Started
## Environment Requirements 
Before conducting the experiment, make sure you have installed the latest Conda environment. Use the following command to create the environment required for the experiment.
```
conda create -n GTMD python=3.9
conda activate GTMD
pip install -r requirements.txt
```
## Begin Training
Before starting the training, please make sure to download the dataset and change the path of the processed dataset to the appropriate location.
You can start the experiment by using the following commands.
use `python -u run.py --\dataset path --required parameters` in the terminal to start training.
