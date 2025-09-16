import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] - %(message)s')

class MOVIC(Dataset):

    def __init__(self, data_path, split='train' ,transforms=None):
        data_directory=os.path.join(data_path, split)
        if not os.path.exists(data_directory):
            if not os.path.exists(os.path.abspath(data_directory)):
                raise Exception("Dataset was not found!")
        
        number_of_frames_per_video=24
        
        self.rgbs = self.collect_files(data_directory, 'rgb*.png', group_size=24)
        self.flows = self.collect_files(data_directory, 'flow*.png', group_size=24)
        self.coords = self.collect_files(data_directory, 'coords*.pt')
        self.masks = self.collect_files(data_directory, 'mask*.pt')

        assert len(self.rgbs) == len(self.flows) == len(self.coords) == len(self.masks), "Data and annotations need to be of the same size"

        logging.info(f"{split.upper()} Data Loaded: Coordinates: {len(self.coords)}, Masks: {len(self.masks)}, RGB videos:  {len(self.rgbs)}, Flows:  {len(self.flows)}")

    def __getitem__(self, idx):
        # return self.coord[idx], self.mask[idx], self.rgb[idx], self.flow[idx]
 
        rgb_paths = self.rgbs[idx]  # list of frame paths for the video frames = idx
        flow_paths = self.flows[idx]
          
        rgbs =  torch.stack([io.read_image(p).to(torch.float32) for p in rgb_paths])
        flows =  torch.stack([io.read_image(p).to(torch.float32) for p in flow_paths])
        coords = torch.load(self.coords[idx], map_location="cpu")  # loaded lazily
        masks = torch.load(self.masks[idx], map_location="cpu")
        
        return rgbs, masks, flows, coords
    
    def collect_files(self, data_directory, condition, group_size=None):
        """
        Collect files matching a pattern and optionally group them by video.
        """
        files = sorted(glob.glob(os.path.join(data_directory, condition)))
        
        if group_size:  # group into videos
            files = [files[i:i+group_size] for i in range(0, len(files), group_size)]
        
        return files


    def __len__(self):
        return len(self.masks)


    def get_video_frame_labels(self, com, bbox, masks, rgbs, flows):
        '''
        outputs 24 entries, each of which corresponds to data of a frame data.
        '''
        output=[]
        for i in range(com.shape[0]):
            output.append((com[i], bbox[i], masks[i], rgbs[i], flows[i]))
        
        return output 