# Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning

This is the official repo of the paper [Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient
Crossmodal Learning](https://arxiv.org/abs/2303.12745) published at ICCV 2023. DOLOS is a online reality-TV gameshow based deception dataset, which can be used for multimodal deception detection research.

## DOLOS

+ Dolos.xlsx - Original Dolos dataset with MUMIN annotations

+ dolos_timestamps(.txt/.csv) - use this file to download the Dolos dataset

+ train_dolos.xlsx - this is the processed annotations for dolos dataset. Use this file for multi-task learning.

## MODEL

+ Dolos_base - PyTorch scripts to train the model on Dolos dataset

+ training_protocols - 3 training protocols used in the paper
