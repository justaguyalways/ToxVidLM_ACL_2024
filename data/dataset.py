import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import random
import torchaudio
import pytorchvideo.data
from .utils import return_audio_tensor, return_image_tensor, return_video_tensor, prepare_batch
from ast import literal_eval
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
    CenterCrop,
    Normalize,
    RandomRotation,
)
            

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, train):
        self.dataframe = dataframe
        self.train = train
        self.tokenizer = tokenizer
        
        if self.train:
            self.video_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop((224,224)),
                    RandomHorizontalFlip(p=0.5),
                ]
            )
        else:
            self.video_transform = Compose(
                [
                    Lambda(lambda x: x / 255.0),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
            

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        
        dialogue = self.dataframe["text"].iloc[idx] #change column name in dataframe
        offensive_labels = torch.tensor(literal_eval(self.dataframe["offensive"].iloc[idx]), device="cpu")
        offensive_level_labels = torch.tensor(literal_eval(self.dataframe["offensiveness level"].iloc[idx]), device="cpu")
        sentiment_labels = torch.tensor(literal_eval(self.dataframe["sentiment"].iloc[idx]), device="cpu")
        
        video = self.dataframe['video_path'].iloc[idx]
        audio = self.dataframe['audio_path'].iloc[idx]
        
        audio = return_audio_tensor(audio)
        video = return_video_tensor(video)
                
        video = self.video_transform(video)

        sample = {
            'dialogue' : dialogue,
            'offensive' : offensive_labels,
            'offensive_level' : offensive_level_labels,
            'sentiment' : sentiment_labels,
            'video': video,
            'audio': audio,
        }
        
        sample = prepare_batch(batch=sample, tokenizer=self.tokenizer)

        return sample