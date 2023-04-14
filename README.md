# Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning

This repo provides a PyTorch implementation of the paper [Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient
Crossmodal Learning](https://arxiv.org/abs/2303.12745)

## MODEL

+ Dolos_base - PyTorch scripts to train the model on Dolos dataset

+ training_protocols - 3 training protocols used in the paper


## DATA

+ The scripts for downloading the Dolos dataset and data preprocesing can be found here https://github.com/NMS05/AV-Data-Processing 

+ Dolos.xlsx - Original Dolos dataset with MUMIN annotations

+ dolos_timestamps(.txt/.csv) - use this file to download the Dolos dataset

+ train_dolos.xlsx - this is the processed annotations for dolos dataset. Use this file for multi-task learning.
