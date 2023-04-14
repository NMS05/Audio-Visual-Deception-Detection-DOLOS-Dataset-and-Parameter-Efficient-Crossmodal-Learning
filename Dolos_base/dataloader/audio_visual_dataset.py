import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchaudio


class AudioVisualDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, img_dir, num_tokens=64, frame_size=160):
        super(AudioVisualDataset, self).__init__()

        self.annos = pd.read_csv(annotations_file)  # [file_name, label, gender]
        self.audio_dir = audio_dir  # all files in '.wav' format
        self.num_tokens = num_tokens

        self.img_dir = img_dir
        self.frame_size = frame_size  # 224 or any multiple of 32
        self.transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Resize(frame_size),
            # normalize to imagenet mean and std values
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.number_of_target_frames = num_tokens

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        # select one clip
        clip_name = self.annos.iloc[idx, 0]

        # get audio path
        audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
        # load the audio file with torch audio
        waveform, sample_rate = torchaudio.load(audio_path)
        # use mono audio instead os stereo audio (use left by default)
        waveform = waveform[0]

        # calculate duration of the audio clip
        clip_duration = len(waveform) / sample_rate
        """
        # for wav2vec2, 1 Token corresponds to ~ 321.89 discrete samples
        # to get precisely 64 tokens (a hyperparameter that can be changed), the length of input discrete samples to the model should be 321.89 * 64
        # divide the above by the clip duration to get new sample rate (or) new_sample_rate * clip_duration = 321.89 * num tokens
        """
        new_sample_rate = int(321.893491124260 * self.num_tokens / clip_duration)
        # resample
        waveform = torchaudio.functional.resample(waveform, sample_rate, new_sample_rate)
        # required by collate function
        mono_waveform = waveform.unsqueeze(0)
        mono_waveform.type(torch.float32)

        #  get face feature path
        file_path = self.img_dir + clip_name + '/'
        # list all jpeg images
        frame_names = [i for i in os.listdir(file_path) if i.split('.')[-1] == 'jpg']

        # sample 64 face frames
        target_frames = np.linspace(0, len(frame_names) - 1, num=self.number_of_target_frames)
        target_frames = np.around(target_frames).astype(
            int)  # certain frames may be redundant because of rounding to the nearest integer
        face_frames = []
        for i in target_frames:
            img = np.asarray(Image.open(file_path + frame_names[i])) / 255.0
            face_frames.append(self.transforms(img))
        face_frames = torch.stack(face_frames, 0)
        face_frames.type(torch.float32)

        # assign integer to labels
        str_label = self.annos.iloc[idx, 1]
        if str_label == 'truth' or str_label == 'truthful' or str_label == 0:
            label = 0
        elif str_label == 'deception' or str_label == 'lie' or str_label == 1:
            label = 1
        # undefined label error
        else:
            raise Exception("undefined label")

        return mono_waveform, face_frames, label


def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def af_collate_fn(batch):
    tensors, face_tensors, targets = [], [], []

    # Gather in lists, and encode labels as indices
    for waveform, face_frames, label in batch:
        tensors += [waveform]
        face_tensors += [face_frames]
        targets += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    tensors = af_pad_sequence(tensors)
    face_tensors = torch.stack(face_tensors)
    targets = torch.stack(targets)

    return tensors, face_tensors, targets
