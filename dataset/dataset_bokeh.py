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

        self._source_paths = sorted(glob.glob(osp.join(root_folder, "*.src.jpg")))
        self._target_paths = sorted(glob.glob(osp.join(root_folder, "*.tgt.jpg")))
        self._alpha_paths = sorted(glob.glob(osp.join(root_folder, "*.alpha.png")))
        

        self._meta_data = self._read_meta_data(osp.join(root_folder, "meta.txt"))


    def __len__(self):
        return len(self._meta_data)

    def _read_meta_data(self, meta_file_path: str):
        """Read the meta file containing source / target lens and disparity for each image.

        Args:
            meta_file_path (str): File path

        Raises:
            ValueError: File not found.

        Returns:
            dict: Meta dict of tuples like {id: (id, src_lens, tgt_lens, disparity)}.
        """
        if not osp.isfile(meta_file_path):
            raise ValueError(f"Meta file missing under {meta_file_path}.")

        meta = []
        with open(meta_file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            id, src_lens, tgt_lens, disparity = [part.strip() for part in line.split(",")]
            
            id = int(id)
            if src_lens == "Canon50mmf1.4BS":
                src_lens = 1.0
            elif src_lens == "Canon50mmf1.8BS":
                src_lens = 0.75
            elif src_lens == "Sony50mmf1.8BS":
                src_lens = 0.625
            else:
                src_lens = 0.0

            if tgt_lens == "Canon50mmf1.4BS":
                tgt_lens = 1.0
            elif tgt_lens == "Canon50mmf1.8BS":
                tgt_lens = 0.75
            elif tgt_lens == "Sony50mmf1.8BS":
                tgt_lens = 0.625
            else:
                tgt_lens = 0.0

            if id >= 19500:
                id = id-19500
            disparity = int(disparity)/80
            meta.append((id,src_lens, tgt_lens, disparity))
        return meta

    def __getitem__(self, index):
        
        id, src_lens, tgt_lens, disparity = self._meta_data[index]
        
        source = Image.open(self._source_paths[id])
        target = Image.open(self._target_paths[id])
        source_alpha = Image.open(self._alpha_paths[id])
        

        if tgt_lens < src_lens:
            input_cond = np.asarray([tgt_lens,disparity])
            input_cond = np.float32(input_cond)
            input_cond = torch.from_numpy(input_cond)

            output_cond = np.asarray([src_lens-tgt_lens,disparity])
            output_cond = np.float32(output_cond)
            output_cond = torch.from_numpy(output_cond)

            source = self._transform(source)
            target = self._transform(target)
            source_alpha = self._transform(source_alpha)
            
            dis_vector = np.asarray([disparity])
            dis_vector = np.float32(dis_vector)
            dis_vector = torch.from_numpy(dis_vector)

            id = self._transform(np.array([[id]]))

            return {
                "source": target,
                "target": source,
                "source_alpha": source_alpha,
                "input_cond": input_cond,
                "output_cond": output_cond,
                "id": id,
                "disparity":dis_vector,
            }

        else:
            input_cond = np.asarray([src_lens,disparity])
            input_cond = np.float32(input_cond)
            input_cond = torch.from_numpy(input_cond)

            output_cond = np.asarray([tgt_lens-src_lens,disparity])
            output_cond = np.float32(output_cond)
            output_cond = torch.from_numpy(output_cond)
            
            id = self._transform(np.array([[id]]))
            source = self._transform(source)
            target = self._transform(target)
            source_alpha = self._transform(source_alpha)
            
            dis_vector = np.asarray([disparity])
            dis_vector = np.float32(dis_vector)
            dis_vector = torch.from_numpy(dis_vector)

            return {
                "source": source,
                "target": target,
                "source_alpha": source_alpha,
                "input_cond": input_cond,
                "output_cond": output_cond,
                "id": id,
                "disparity":dis_vector,
            }
