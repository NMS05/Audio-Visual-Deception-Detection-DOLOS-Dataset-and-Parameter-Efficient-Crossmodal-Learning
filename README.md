# Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning

This is the official repo of the paper [Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient
Crossmodal Learning](https://arxiv.org/abs/2303.12745) published at ICCV 2023. DOLOS is a online reality-TV gameshow based deception dataset, which can be used for multimodal deception detection research.

## DOLOS Dataset

+ Dolos.xlsx - The original Dolos dataset with MUMIN annotations. Every sample contains a YouTube video link to the gameshow, its time stamp and its corresponding label (truth or deception).

+ dolos_timestamps(.txt/.csv) - This file is used to download the Dolos dataset. Run the script [YT_video_downloader.py](https://github.com/NMS05/AV-Data-Processing) and pass the timestamps file as an argument. It will automatically download (only those currently available) from YouTube and save them as .mp4 files.

+ train_dolos.xlsx - This file contains the processed MUMIN annotations for dolos dataset. To be used for multi-task learning.

## Training

+ scripts - contains PyTorch scripts to train the model on Dolos dataset

+ training_protocols - 3 training protocols used in the paper
