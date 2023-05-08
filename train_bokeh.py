import os

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from dataset.dataset_bokeh import BokehDataset
from metrics import calculate_lpips, calculate_psnr, calculate_ssim
from models.CUGAN import CUGAN
from torchvision import transforms
import matplotlib.pyplot as plt
import warnings
from msssim import MSSSIM
warnings.filterwarnings("ignore")
to_image = transforms.Compose([transforms.ToPILImage()])
to_tensor = ToTensor()
to_pil = ToPILImage()

np.random.seed(0)
torch.manual_seed(0)

class CharbonnierLoss(torch.nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,x):
        value=torch.sqrt(torch.pow(x,2)+self.epsilon2)
        return torch.mean(value)


def fake_eval(model, dataloader):
    model.eval()
    test_iter = iter(dataloader)
    lpips = 0
    psnr = 0
    ssim = 0
    total_num = 1*len(test_iter)
    print(total_num)
    for i in range(len(test_iter)):

        batch = next(test_iter)
        id = batch["id"].detach().cpu()
        id = id[0][0][0][0]
        source = batch["source"].cuda()
        target = batch["target"].cuda()

        source_alpha = batch["source_alpha"].cuda()
        
        source = source*(1-source_alpha)
        target = target*(1-source_alpha)

        input_cond = batch["input_cond"].cuda()
        output_cond = batch["output_cond"].cuda()
        
        with torch.no_grad():
            output = model(source,output_cond,output_cond)
            output = output*(1-source_alpha)
            output = torch.clamp(output,min=0.0, max=1.0)
        
        
        # Calculate metrics
        lpips += np.mean([calculate_lpips(img0, img1) for img0, img1 in zip(output, target)])
        psnr += np.mean(
            [calculate_psnr(np.asarray(to_pil(img0)), np.asarray(to_pil(img1))) for img0, img1 in zip(output, target)]
        )
        ssim += np.mean(
            [calculate_ssim(np.asarray(to_pil(img0)), np.asarray(to_pil(img1))) for img0, img1 in zip(output, target)]
        )
    
    lpips = lpips/total_num
    psnr = psnr/total_num
    ssim = ssim/total_num
    model.train()

    print(f"Metrics: lpips={lpips:0.05f}, psnr={psnr:0.05f}, ssim={ssim:0.05f}")

    return psnr


def train():
    os.makedirs("outputs", exist_ok=True)

    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    print("CUDA visible devices: " + str(torch.cuda.device_count()))
    print("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))

    model = CUGAN(in_nc=3,out_nc=3,cond_dim=2,stages_blocks_num=[2,2,2],stages_channels=[32,64,128],downSample_Ksize=2).to(device)
    model = torch.nn.DataParallel(model.cuda())
    model.train()

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',factor=0.5,patience=5,verbose=True)
    #criterion = nn.L1Loss()
    #criterion = CharbonnierLoss()
    MS_SSIM = MSSSIM()
    criterion = nn.MSELoss()

    train_dataset = BokehDataset("data/train", transform=ToTensor())
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4,pin_memory=True)

    test_dataset = BokehDataset("data/train_val", transform=ToTensor())
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True,num_workers=1,pin_memory=True)

    best_psnr = 0
    for epoch in range(500):
        train_iter = iter(train_dataloader)
        for i in range(len(train_iter)):
            
            batch = next(train_iter)
            source = batch["source"].cuda()
            target = batch["target"].cuda()

            source_alpha = batch["source_alpha"].cuda()
            
            source = source*(1-source_alpha)
            target = target*(1-source_alpha)

            output_cond = batch["output_cond"].cuda()
            
            optimizer.zero_grad()
            output = model(source,output_cond,output_cond)
            output = output*(1-source_alpha)
            loss_ssim = MS_SSIM(output,target)
            loss = criterion(output,target) + (1-loss_ssim)*0.25
            loss.backward()
            optimizer.step()

            if (i+1)%500 == 0:
                print(f"Epoch {epoch} checkpoint, loss: {loss.item():0.04f}.")
                
        psnr = fake_eval(model, test_dataloader)
        if psnr>=best_psnr:
            best_psnr = psnr

        print("best_psnr",best_psnr)
        scheduler.step(psnr)
        model.eval().cpu()   
        torch.save(model.state_dict(), "saved_models/"+"bokeh128_epoch"+str(epoch)+".pth")
        model.to(device).train()



if __name__ == "__main__":
    train()