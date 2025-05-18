
from utils import *
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
import re
from transformations import get_train_transforms


class KTHActionDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, max_frames=10, img_size=(64, 64)):
        self.root_dir = root_dir
        self.transform = transform
        self.max_frames = max_frames
        self.img_size = img_size
        self.categories = ["walking", "jogging", "running", "boxing", "handwaving", "handclapping"]
        self.label_map = {cat: i for i, cat in enumerate(self.categories)}
        
        # Collect frame directories and labels
        self.frame_dirs = []
        self.labels = []
        for category in self.categories:
            category_dir = os.path.join(root_dir, category)
            if not os.path.exists(category_dir):
                continue
            for seq_dir in os.listdir(category_dir):
                if not os.path.isdir(os.path.join(category_dir, seq_dir)):
                    continue  # Skip files like test_keypoints64x64.json
                # Extract person ID (e.g., 01 from person01_handwaving_d3)
                match = re.search(r"person(\d+)", seq_dir)
                if not match:
                    continue
                person_id = int(match.group(1))
                # Split based on person ID. 0-16 Train, 17-25 Test
                if (split == "train" and person_id <= 16) or (split == "test" and person_id > 16):
                    self.frame_dirs.append(os.path.join(category_dir, seq_dir))
                    self.labels.append(self.label_map[category])

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, idx):
        # Get sequence directory and label
        seq_dir = self.frame_dirs[idx]
        label = self.labels[idx]
        
        # Load frames
        frame_files = sorted([f for f in os.listdir(seq_dir) if f.endswith(('.jpg', '.png'))]) #sort files in ascending order
        # print(frame_files)
        frames = []

        '''Ensuring that everytime we get a new sequnce of frames with len=self.max_frames 
        in the choosen seq_dir
        '''
        total_frames = len(os.listdir(seq_dir))
        start_idx = random.randint(0, total_frames-self.max_frames)
        frame_files = frame_files[start_idx : start_idx+self.max_frames]

        for frame in frame_files:
            print(frame)
            frame_path = os.path.join(seq_dir, frame)
            frame = Image.open(frame_path).convert('L')  # Grayscale
            frame = frame.resize(self.img_size)
            frame = np.array(frame, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            frames.append(frame)
        
        # Pad to fixed length
        if len(frames) < self.max_frames:
            last_frame = frames[-1] 
            frames.extend([last_frame] * (self.max_frames - len(frames))) # add missing frames to match our desired len
        frames = frames[:self.max_frames]
        
        # Convert to tensor
        frames = torch.tensor(np.array(frames), dtype=torch.float32).unsqueeze(1)  # Shape: [T, 1, H, W]
        # print(frames.shape)
        
        # Applying transforms
        if self.transform:
            frames = self.transform(frames)
        # print(frames.shape)

        return frames, label
