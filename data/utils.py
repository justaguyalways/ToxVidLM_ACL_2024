import av
import numpy as np
import os
from transformers import AutoImageProcessor, CLIPProcessor, AutoFeatureExtractor, WhisperFeatureExtractor
import cv2
import random
import torch

np.random.seed(0)

max_length = 50000
max_len = 64

audio_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")

def get_actual_seg_len(container):
    length = 0
    for i, frame in enumerate(container.decode(video=0)):
        length+=1
    return length
    
    
def read_video_pyav(container, indices, clip_len, seg_len):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    if seg_len > clip_len:
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
    else:
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        
        start_frame = 0
                
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
                if i == 0:
                    start_frame = frame
        
        init_frames = [start_frame]*(clip_len-seg_len)
        init_frames.extend(frames)
        frames = init_frames
       
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    
    if seg_len > clip_len: 
        converted_len = int(clip_len * frame_sample_rate)

        # Ensure the chosen segment is valid
        start_idx = np.random.randint(low=0, high=seg_len - converted_len + 1)
        end_idx = start_idx + converted_len

        # Generate equally spaced indices within the chosen segment
        indices = np.linspace(start_idx, end_idx - 1, num=clip_len, dtype=np.int64)
    
    else:
        indices_start = [0]*(clip_len-seg_len)
        indices = range(seg_len)
        indices_start.extend(indices)
        indices = np.array(indices_start, dtype=np.int64)
    
    # print(indices.shape)
    return indices

def read_random_jpg(directory_path):
    jpg_files = [file for file in os.listdir(directory_path) if file.endswith('.jpg')]

    if not jpg_files:
        raise FileNotFoundError("No '.jpg' files found in the specified directory.")

    selected_file = random.choice(jpg_files)
    file_path = os.path.join(directory_path, selected_file)

    image = cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)

    return image 

def return_video_tensor(file_path, clip_len=16, frame_sample_rate=5):
    container = av.open(file_path)
    
    # seg_len=container.streams.video[0].frames
    seg_len = get_actual_seg_len(container)
    frame_sample_rate_calculated=int(seg_len/clip_len)
    if frame_sample_rate_calculated < frame_sample_rate:
        frame_sample_rate=frame_sample_rate_calculated-1
    if not frame_sample_rate:
        frame_sample_rate=1
    indices = sample_frame_indices(clip_len=clip_len, frame_sample_rate=frame_sample_rate, seg_len=seg_len)
    video = read_video_pyav(container, indices, clip_len, seg_len)
    inputs = video_processor(list(video), return_tensors="pt")
    inputs = inputs["pixel_values"] #shape - [1, 16, 3, 224, 224]
    
    # if(inputs.shape[1] != 16):
        # print(indices, frame_sample_rate, seg_len, inputs.shape[1])
    return inputs.squeeze(dim=0).cpu()


def return_image_tensor(directory_path):
    image = read_random_jpg(directory_path=directory_path)
    inputs = image_processor(images = image, return_tensors="pt")
    inputs = inputs["pixel_values"]  #shape - [1, 3, 224, 224]
    return inputs.squeeze(dim=0).cpu()

def return_audio_tensor(file_path):
    input_audio = np.load(file_path)
    input_audio = audio_processor(input_audio, sampling_rate=16000, return_tensors="pt")
    input_features = input_audio["input_features"]
    # input_audio = torch.tensor(input_audio)
    # inputs = audio_processor(input_audio, sampling_rate=16000, max_length=max_length, truncation=True, return_tensors="pt")
    # total_padding = max(0, max_length - len(input_audio))
    # inputs["input_values"] = torch.nn.functional.pad(inputs["input_values"], (0, total_padding), value=0.0).squeeze(dim=0).cpu()
    # inputs["attention_mask"] = torch.nn.functional.pad(inputs["attention_mask"], (0, total_padding), value=0.0).squeeze(dim=0).cpu()
    return input_features.squeeze(dim=0).cpu()

def prepare_batch(batch, tokenizer):
    dialogue = batch["dialogue"]
    tokenized_text = tokenizer(dialogue, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len-3, add_special_tokens=False)
    token_type_ids = torch.ones_like(tokenized_text.input_ids)
    
    new_batch = {  
        'offensive': batch['offensive'],
        'offensive_level': batch['offensive_level'],
        'sentiment': batch['sentiment'],
        'video': batch['video'],
        'audio': batch['audio'],
        'input_ids': tokenized_text.input_ids.squeeze(dim=0),
        'attention_mask': tokenized_text.attention_mask.squeeze(dim=0),
        'token_type_ids': token_type_ids.squeeze(dim=0),
    }
    
    return new_batch