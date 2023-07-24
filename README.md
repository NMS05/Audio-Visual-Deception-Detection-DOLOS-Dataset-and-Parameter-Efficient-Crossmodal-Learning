# Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning

This is the official repo of the paper [Audio-Visual Deception Detection: DOLOS Dataset and Parameter-Efficient
Crossmodal Learning](https://arxiv.org/abs/2303.12745) published at ICCV 2023.

# DOLOS Dataset

<img src="https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/blob/main/imgs/data_dist.png" width="800" height="450"> 

**DOLOS** is a online reality-TV gameshow based deception dataset curated for multimodal deception detection research. The DOLOS dataset can be downloaded from [ROSE Lab, NTU](https://rose1.ntu.edu.sg/). The dataset is organized as follows,

+ **Dolos.xlsx** - This spreadsheet contains the original and unprocessed DOLOS dataset with MUMIN annotations. Every sample (row) contains the following information
    - YouTube video link to the gameshow, time stamps and its corresponding label (truth or deception).
    - Metadata: file_name, subject_name, subject_gender
    - MUMIN annotations as nested list of time intervals

+ **dolos_timestamps(.txt/.csv)** - This file can be used to download the Dolos dataset. Run the script [YT_video_downloader.py](https://github.com/NMS05/AV-Data-Processing) and pass this timestamps file as an argument. It will automatically download the videos (only those currently available) from YouTube and save them as .mp4 files.

+ **train_dolos.xlsx** - This file contains the MUMIN features as binary annotations, which can be used for multi-task learning. This file is also suited for PyTorch DataLoader.

+ **Training_Protocols/** - This folder contains the training protocols used for experiments in the paper.
  - train_fold.csv/test_fold.csv = used for 3-fold evaluation of multimodal deception detector
  - long.csv/short.csv = to investigate the perfomance with respect to variability in speaking (deception) duration
  - male.csv/female.csv = protocol to investigate the influence of gender in deception detector

# Training

<img src="https://github.com/NMS05/Audio-Visual-Deception-Detection-DOLOS-Dataset-and-Parameter-Efficient-Crossmodal-Learning/blob/main/imgs/pecl.png" width="850" height="400"> 

+ **Data Pre-Processing** - Run [extract_face_frames.py and video_to_audio.py](https://github.com/NMS05/AV-Data-Processing) to extract the RGB face frames and '.wav' audio files from the downloaded videos.
+ **scripts/** - Contains the PyTorch code to train the model on the pre-processed Dolos dataset. The scripts are organized as follows,
    - dataloader/
        - audio_visual_dataset.py = Requires '.csv' files from Training_Protocols. Upon call, it returns the (audio wavefrom, face frames, labels) as a torch tensor.
    - models/
        - adapter.py = Contains the NLP adapter and Uniform Temporal Adapter (UTA) for Wav2Vec2 and ViT model.
        - audio_model.py = Wav2Vec2 model for audio (speech) deception detection.
        - visual_model.py = ViT-B16 model for visual (facial) deception detection.
        - fusion_model.py = (Wav2Vec2 + ViT-B16) with Plug-in Audio-Visual Fusion (PAVF) for multimodal deception detection.
    - train_test.py = script to train model(s) for unimodal and multimodal deception detection.
    - run **train_fusion.sh** for reproducibility. 
