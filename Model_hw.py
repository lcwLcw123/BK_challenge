import os

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
import math
from models.CUGAN import CUGAN
from models.CUGAN_stack import CUGAN_stack
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings
def get_gaussian_kernel(kernel_size=9, sigma=2, channels=3):
# Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
    torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) /\
    (2*variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter

class Model_load(nn.Module):
    def __init__(self):
        super(Model_load, self).__init__()
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        print("CUDA visible devices: " + str(torch.cuda.device_count()))
        print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

        self.model_bokeh_1000 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
       
        self.model_bokeh_1000 = torch.nn.DataParallel(self.model_bokeh_1000.cuda())
        
        self.checkpoint = torch.load("saved_models/model_hw.pt")

        self.blur_layer = get_gaussian_kernel(21,5,3).to(device)

#-------------------------------------------------------------------------------------------------------------------

        self.model_bokeh_1000.load_state_dict(self.checkpoint['model_path_bokeh_1000'],strict=True)

        print('already load model')

        self.model_bokeh_1000.eval()


#---------------------------------------------------------------------------------------------------------
    def inference(self,batch):

        source = batch["source"].cuda()
        source_alpha = batch["source_alpha"].cuda()

        source_output = source.clone()

        source = self.blur_layer(source)
        source = source*(1-source_alpha)

        output_cond = batch["output_cond"].cuda()
        
        extent = output_cond[0][0].item()
   
        with torch.no_grad():
            output = self.model_bokeh_1000(source,output_cond,output_cond)
            output = output*(1-source_alpha)+source_output*source_alpha
            output = torch.clamp(output,min=0.0, max=1.0)
                    
        return output



torch.backends.cudnn.deterministic = True
device = torch.device("cuda")
print("CUDA visible devices: " + str(torch.cuda.device_count()))
print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))


model_bokeh_1000 = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
model_bokeh_1000 = torch.nn.DataParallel(model_bokeh_1000.cuda())

model_path_bokeh_1000 = "/home/chenzigeng/dehaze/NTIRE23BokehTransformation/examples/modelzoo_mse/bokeh128_mse_1.0_3_21_epoch0.pth"
#----------------------------------------------------------------------------------------------------------------------------------

PATH = "saved_models/model_hw.pt"
# model = Model() 
torch.save({
            'model_path_bokeh_1000': torch.load(model_path_bokeh_1000),
            }, PATH)

print('save ok!')
model2 = Model_load()