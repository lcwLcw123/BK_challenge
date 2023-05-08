import glob
import os.path as osp
# from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.train import parse_options1,parse_options2
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite


from torchvision.transforms import ToPILImage, ToTensor
import cv2
import os
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from dataset.dataset_final import BokehDataset
from models.CUGAN import CUGAN
from models.CUGAN_stack import CUGAN_stack
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings
from Model_hw import Model_load
import math
from PIL import ImageFilter
warnings.filterwarnings("ignore")
to_image = transforms.Compose([transforms.ToPILImage()])
to_tensor = ToTensor()
to_pil = ToPILImage()

np.random.seed(0)
torch.manual_seed(0)

def train():
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))
    model = Model_load()
    model.eval()
    
    test_dataset = BokehDataset("huawei_task/huawei_image", transform=ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=2)

    test_iter = iter(test_dataloader)
    total_num = 1*len(test_iter)

    print("total_num:",total_num)
 
    for i in tqdm(range(total_num)):

        batch = next(test_iter)
        
        output_cond = batch["output_cond"].cuda()
        
        id = batch["id"]

        
        with torch.no_grad():

            output = model.inference(batch)


        output = np.array(to_image(torch.squeeze(output.float().detach().cpu())))
        output = output[:,:,(2,1,0)]
        cv2.imwrite("./result/"+id[0]+".jpg", output)

#
if __name__ == "__main__":
    train()