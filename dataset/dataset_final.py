import glob
import os.path as osp
from typing import Callable, Optional
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class BokehDataset(Dataset):
    def __init__(self, root_folder: str, transform: Optional[Callable] = None):

        self._root_folder = root_folder
        self._transform = transform

        self._source_paths = sorted(glob.glob(osp.join(root_folder, "*.jpg")))
        self._source_alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.mask.png")))

    def __len__(self):
        return len(self._source_paths)


    def __getitem__(self, index):
        
        source = Image.open(self._source_paths[index])
        source_alpha = Image.open(self._source_alpha_paths[index])

        filename = osp.basename(self._source_paths[index])
        id = filename.split(".")[0]

        source = self._transform(source)
        source_alpha = self._transform(source_alpha)
        
        output_cond = np.asarray([1.0,1.0])
        output_cond = np.float32(output_cond)
        output_cond = torch.from_numpy(output_cond)

        return {
            "source": source,
            "source_alpha": source_alpha,
            "id": id,
            "output_cond": output_cond
        }

