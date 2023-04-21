# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 21:16:33 2023

@author: johan
"""



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import utils,models
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from PIL import Image
import PIL
from tqdm import trange
from time import sleep
from scipy.io import loadmat
import torchvision.datasets as dset
from torch.utils.data import sampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from partialconv2d_S2S import PartialConv2d
from layer import *
from torch.nn import init, ReflectionPad2d, ZeroPad2d
from torch.optim import lr_scheduler
from utils import *
from Functions_pytorch import *
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam


torch.manual_seed(0)

class TwoCon(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = TwoCon(1, 64)
        self.conv2 = TwoCon(64, 64)
        self.conv3 = TwoCon(64, 64)
        self.conv4 = TwoCon(64, 64)  
        self.conv6 = nn.Conv2d(64,1,1)
        

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x = self.conv4(x3)
        x = torch.sigmoid(self.conv6(x))
        return x





def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if gpu_ids:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def image_loader(image, device, p1, p2):
    """load image, returns cuda tensor"""
    loader = T.Compose([T.ToPILImage(),T.RandomHorizontalFlip(torch.round(torch.tensor(p1))),T.RandomVerticalFlip(torch.round(torch.tensor(p2))),T.ToTensor()])
    image = torch.tensor(image).float()
    image= loader(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.to(device)


class Denseg_N2F:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        lam: float = 0.01,
        device: str = 'cuda:0',
        verbose = False,
    ):
        self.learning_rate = learning_rate
        self.lam = lam
        self.sigma_tv = 1/2
        self.tau = 1/4
        self.theta = 1.0
        self.difference = []
        self.p = []
        self.q=[]
        self.r = []
        self.x_tilde = []
        self.device = device
        self.f_std = []
        self.Dice=[]
        self.Dice.append(1)
        self.fid=[]
        self.tv=[]
        self.tv_plot=[]
        self.fidelity_fg = []
        self.fidelity_bg = []
        self.en = []
        self.iteration = 0
        self.f1 = None
        self.f2 = []
        self.verbose = True
        self.Npred = 100
        self.loss_s2s=[]
        self.loss_list_N2F=[]
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.energy_denoising = []
        self.val_loss_list_N2F = []
        self.bg_loss_list = []
        self.number_its_N2F=500
        self.fidelity_fg_d_bg =[]
        self.fidelity_bg_d_fg = []
        self.val_loss_list_N2F_firstmask = []
        self.old_x = 0
        self.p1 = 0
        self.mu = 0
        self.first_mask = 0
        self.val_loss_list_N2F_currentmask = []
        self.current_mask = 0
        self.first_loss = 4
        self.current_loss = torch.tensor(1)
        self.previous_loss = torch.tensor(2)
        self.iteration_index = 0
        self.variance = []
        
    def normalize(self,f): 
        '''normalize image to range [0,1]'''
        f = torch.tensor(f).float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f

        
    def initialize(self,f):
        #prepare input for segmentation
        f = self.normalize(f)
        self.p = gradient(torch.clone(f))
        self.q = torch.clone(f)
        self.r = torch.clone(f)
        self.x_tilde = torch.clone(f)
        self.x = torch.clone(f)
        self.f = torch.clone(f)

 ######################## Classical Chan Vese step############################################       
    def segmentation_step(self,f):
        f_orig = torch.clone(f)
        # for segmentaion process, the input should be normalized and the values should lie in [0,1]
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)

        self.p = p1.clone()
        self.q = q1.clone()
        self.r = r1.clone()
        # Update primal variables
        x_old = torch.clone(self.x)  
        self.x = proj_unitintervall(x_old + self.tau*div(p1) - self.tau*adjoint_der_Fid1(x_old, f, self.q) - self.tau *
                               adjoint_der_Fid2(x_old, f, self.r))  # proximity operator of indicator function on [0,1]
        self.x_tilde = self.x + self.theta*(self.x-x_old)
        if self.verbose == True:
            fidelity = norm1(Fid1(torch.clone(self.x), f)) + norm1(Fid2(torch.clone(self.x),f))
            total = norm1(gradient(torch.clone(self.x)))
            self.fid.append(fidelity.cpu())
            tv_p = norm1(gradient(torch.clone(self.x)))
            self.tv.append(total.cpu())
            energy = fidelity +self.lam* total
            self.en.append(energy.cpu())
          #  plt.plot(np.array(self.tv), label = "TV")
            if self.iteration %999 == 0:  
                plt.plot(np.array(self.en[:]), label = "energy")
                plt.plot(np.array(self.fid[:]), label = "fidelity")
              #  plt.plot(self.lam*np.array(self.tv[498:]))
                plt.legend()
                plt.show()
        self.iteration += 1


    def compute_energy(self,x=None):
        if x == None:
            x = torch.clone(self.x)
        diff1 = torch.clone(self.f-self.f1).float()
        diff2 = torch.clone(self.f - self.mu_r2).float()
        energy = torch.sum((diff1)**2*x)+ torch.sum((diff2)**2*(1-x)) + self.lam*norm1(gradient(torch.clone(x)))
        return energy.cpu()
    
    def compute_fidelity(self,x = None):
        if x == None:
            x = torch.clone(self.x)
        diff1 = torch.clone(self.f-self.f1).float()
        diff2 = torch.clone(self.f - self.mu_r2).float()
        fidelity = torch.sum((diff1)**2*x)+ torch.sum((diff2)**2*(1-x))
        return fidelity.cpu()
##################### accelerated segmentation algorithm bg constant############################
##################### accelerated segmentation algorithm bg constant############################
    def segmentation_step2denoisers_acc_bg_constant(self,f, iterations, gt):
        energy_beginning = self.compute_energy()
        f_orig = torch.clone(f).to(self.device)
        f1 = torch.clone(self.f1)

        # compute difference between noisy input and denoised image
        diff1 = torch.clone(f_orig-f1).float()
        #compute difference between constant of background and originial noisy image
        diff2 = torch.clone(f_orig - self.mu_r2).float()


        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)
        self.q = q1.clone()
        self.r = r1.clone()
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        for i in range(iterations):
            self.en.append(self.compute_energy())

            p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
    
            # Fidelity term without norm (change this in funciton.py)
            #self.p = p1.clone()
            self.p = p1.clone()
            # Update primal variables
            self.x_old = torch.clone(self.x)  
    
    
            # constant difference term
            #filteing for smoother differences between denoised images and noisy input images
            self.x = proj_unitintervall(self.x_old + self.tau*div(p1) - self.tau*((diff1)**2) +  self.tau*((diff2**2))) # proximity operator of indicator function on [0,1]
            ######acceleration variables
            self.theta=1/np.sqrt(1+2*self.tau*self.mu)
            self.tau=self.theta*self.tau
            self.sigma_tv = self.sigma_tv/self.theta
            ###### 
            self.x_tilde = self.x + self.theta*(self.x-self.x_old)
           # self.x = torch.round(self.x)
            if self.verbose == True:
                fidelity = self.compute_fidelity()
                fid_den = torch.sum((diff1)**2*self.x)
                fid_fg_denoiser_bg = (torch.sum((diff1)**2*(1-self.x))).cpu()
                fid_bg_denoiser_fg = (torch.sum((diff2)**2*(self.x))).cpu()
                self.fidelity_bg_d_fg.append(fid_bg_denoiser_fg)
                self.fidelity_fg_d_bg.append(fid_fg_denoiser_bg)
                self.fidelity_fg.append(fid_den.cpu())
                #self.difference.append(diff1-diff2)
                fid_const =( torch.sum((diff2**2*(1-self.x)))).cpu()
                self.fidelity_bg.append(fid_const)
                total = norm1(gradient(self.x))
                self.fid.append(fidelity.cpu())
                tv_p = norm1(gradient(self.x))
                tv_pf = norm1(gradient(self.x*f_orig))
                self.tv.append(total.cpu())
                energy = fidelity + self.lam*tv_p
                #self.en.append(energy.cpu())
                self.en.append(self.compute_energy())

                
                gt_bin = torch.clone(gt)
                gt_bin[gt_bin > 1] = 1
                seg = torch.round(torch.clone(self.x))
                fp =torch.sum(seg*(1-gt_bin))
                fn = torch.sum((1-seg)*gt_bin)
                tp = torch.sum(seg*gt_bin)
                tn = torch.sum((1-seg)*(1-gt_bin))
                dice = 2*tp/(2*tp + fn + fp)    
                self.Dice.append(((1-dice)*100).cpu())
              #  plt.plot(np.array(self.tv), label = "TV")
                if self.iteration %5999 == 1:  
                    plt.plot(np.array(self.fidelity_fg), label = "forground_loss")
                    plt.plot(np.array(self.fidelity_bg[:]), label = "background_loss")
                    plt.plot(np.array(self.en[:]), label = "energy")
                    plt.plot(np.array(self.fid[:]), label = "fidelity")
                    plt.plot(np.array(self.tv), label = "TV")
    
                  #  plt.plot(self.lam*np.array(self.tv[498:]))
                    plt.legend()
                    plt.show()







    def denoising_step_r2(self):
        f = torch.clone(self.f)
        self.mu_r2 = torch.sum(f*(1-self.x))/torch.sum(1-self.x)
        

    
    def reinitialize_network(self):
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def N2Fstep(self):
        self.previous_loss = torch.clone(self.current_loss)

        if self.f1 == None:
            self.f1 = torch.clone(self.f)
        f = torch.clone(self.f)
        loss_mask=torch.round(torch.clone(self.x)).detach()
        img = f[0].cpu().numpy()#*loss_mask[0].cpu().numpy()
        img = np.expand_dims(img,axis=0)
        img = np.expand_dims(img, axis=0)
        
        img_test = f[0].cpu().numpy()
        img_test = np.expand_dims(img_test,axis=0)
        img_test  = np.expand_dims(img_test, axis=0)
        
        minner = np.min(img)
        img = img -  minner
        maxer = np.max(img)
        img = img/ maxer
        img = img.astype(np.float32)
        img = img[0,0]
        
        minner_test = np.min(img_test)
        img_test = img_test -  minner_test
        maxer_test = np.max(img_test)
        img_test = img_test/ maxer
        img_test = img_test.astype(np.float32)
        img_test = img_test[0,0]

        shape = img.shape

         
        listimgH_mask = []
        listimgH = []
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
            Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        imgM = loss_mask[0,:Zshape[0],:Zshape[1]]
        
        imgin = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2 = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
                 
        imgin_mask = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        imgin2_mask = np.zeros((Zshape[0]//2,Zshape[1]),dtype=np.float32)
        for i in range(imgin.shape[0]):
            for j in range(imgin.shape[1]):
                if j % 2 == 0:
                    imgin[i,j] = imgZ[2*i+1,j]
                    imgin2[i,j] = imgZ[2*i,j]
                    imgin_mask[i,j] = imgM[2*i+1,j]
                    imgin2_mask[i,j] = imgM[2*i,j]
                if j % 2 == 1:
                    imgin[i,j] = imgZ[2*i,j]
                    imgin2[i,j] = imgZ[2*i+1,j]
                    imgin_mask[i,j] = imgM[2*i,j]
                    imgin2_mask[i,j] = imgM[2*i+1,j]
        imgin = torch.from_numpy(imgin)
        imgin = torch.unsqueeze(imgin,0)
        imgin = torch.unsqueeze(imgin,0)
        imgin = imgin.to(self.device)
        imgin2 = torch.from_numpy(imgin2)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = torch.unsqueeze(imgin2,0)
        imgin2 = imgin2.to(self.device)
        listimgH.append(imgin)
        listimgH.append(imgin2)
        
        
        imgin_mask = torch.from_numpy(imgin_mask)
        imgin_mask = torch.unsqueeze(imgin_mask,0)
        imgin_mask = torch.unsqueeze(imgin_mask,0)
        imgin_mask = imgin_mask.to(self.device)
        imgin2_mask = torch.from_numpy(imgin2_mask)
        imgin2_mask = torch.unsqueeze(imgin2_mask,0)
        imgin2_mask = torch.unsqueeze(imgin2_mask,0)
        imgin2_mask = imgin2_mask.to(self.device)
        listimgH_mask.append(imgin_mask)
        listimgH_mask.append(imgin2_mask)        
         
        listimgV = []
        listimgV_mask=[]
        Zshape = [shape[0],shape[1]]
        if shape[0] % 2 == 1:
            Zshape[0] -= 1
        if shape[1] % 2 == 1:
             Zshape[1] -=1  
        imgZ = img[:Zshape[0],:Zshape[1]]
        imgM = loss_mask[0,:Zshape[0],:Zshape[1]]

         
        imgin3 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4 = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin3_mask = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        imgin4_mask = np.zeros((Zshape[0],Zshape[1]//2),dtype=np.float32)
        for i in range(imgin3.shape[0]):
            for j in range(imgin3.shape[1]):
                if i % 2 == 0:
                    imgin3[i,j] = imgZ[i,2*j+1]
                    imgin4[i,j] = imgZ[i, 2*j]
                    imgin3_mask[i,j] = imgM[i,2*j+1]
                    imgin4_mask[i,j] = imgM[i, 2*j]
                if i % 2 == 1:
                    imgin3[i,j] = imgZ[i,2*j]
                    imgin4[i,j] = imgZ[i,2*j+1]
                    imgin3_mask[i,j] = imgM[i,2*j]
                    imgin4_mask[i,j] = imgM[i,2*j+1]
        imgin3 = torch.from_numpy(imgin3)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = torch.unsqueeze(imgin3,0)
        imgin3 = imgin3.to(self.device)
        imgin4 = torch.from_numpy(imgin4)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = torch.unsqueeze(imgin4,0)
        imgin4 = imgin4.to(self.device)
        listimgV.append(imgin3)
        listimgV.append(imgin4)
        
        imgin3_mask = torch.from_numpy(imgin3_mask)
        imgin3_mask = torch.unsqueeze(imgin3_mask,0)
        imgin3_mask = torch.unsqueeze(imgin3_mask,0)
        imgin3_mask = imgin3_mask.to(self.device)
        imgin4_mask = torch.from_numpy(imgin4_mask)
        imgin4_mask = torch.unsqueeze(imgin4_mask,0)
        imgin4_mask = torch.unsqueeze(imgin4_mask,0)
        imgin4_mask = imgin4_mask.to(self.device)
        listimgV_mask.append(imgin3_mask)
        listimgV_mask.append(imgin4_mask)        
        

        img = torch.from_numpy(img)
     
        img = torch.unsqueeze(img,0)
        img = torch.unsqueeze(img,0)
        img = img.to(self.device)
         
        listimgV1 = [[listimgV[0],listimgV[1]]]
        listimgV2 = [[listimgV[1],listimgV[0]]]
        listimgH1 = [[listimgH[1],listimgH[0]]]
        listimgH2 = [[listimgH[0],listimgH[1]]]
        listimg = listimgH1+listimgH2+listimgV1+listimgV2
         
        listimgV1_mask = [[listimgV_mask[0],listimgV_mask[1]]]
        listimgV2_mask = [[listimgV_mask[1],listimgV_mask[0]]]
        listimgH1_mask = [[listimgH_mask[1],listimgH_mask[0]]]
        listimgH2_mask = [[listimgH_mask[0],listimgH_mask[1]]]
        listimg_mask = listimgH1_mask+listimgH2_mask+listimgV1_mask+listimgV2_mask
        
        img_test = torch.from_numpy(img_test)
        img_test = torch.unsqueeze(img_test,0)
        img_test = torch.unsqueeze(img_test,0)
        img_test = img_test.to(self.device)
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*11
        last10psnr = [0]*105
        cleaned = 0
       # if  self.first_loss > self.current_loss:
        while timesince < self.number_its_N2F:
            indx = np.random.randint(0,len(listimg))
            data = listimg[indx]
            data_mask = listimg_mask[indx]
            inputs = data[0]
            labello = data[1]
            loss_mask = data_mask[1]
            
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss1 = torch.sum((outputs-labello)**2*loss_mask)#+ torch.sum(torch.min(self.f1)-torch.clip(outputs,max=torch.min(self.f1)))#+ 0.1*torch.sum((outputs -  torch.mean(outputs))**2)#/torch.sum(loss_mask)
            loss1.backward()
            self.optimizer.step()
            running_loss1+=loss1.item()
            self.loss_list_N2F.append(loss1.detach().cpu())


            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned)
                outputstest = self.net(img_test.detach()).detach()

              #  self.en.append((torch.sum((outputstest[0]-img_test[0])**2*self.x) + torch.sum((img_test[0] - torch.sum(img_test[0]*(1-self.x))/torch.sum(1-self.x))**2*(1-self.x)) + self.lam*norm1(gradient(self.x))).cpu())
                #self.Dice.append(self.Dice[-1])
                self.val_loss_list_N2F_firstmask.append((torch.sum((outputstest[0]-img_test[0])**2*self.first_mask)/torch.sum(self.first_mask)).cpu())#
                self.current_loss = (torch.sum((outputstest[0]-img_test[0])**2*self.first_mask)/torch.sum(self.first_mask)).cpu()
                self.val_loss_list_N2F_currentmask.append((torch.sum((outputstest[0]-img_test[0])**2*self.x)/torch.sum(self.x)).cpu())
                #self.current_loss = (torch.sum((outputstest[0]-img_test[0])**2*self.x)/torch.sum(self.x)).cpu()
                # compute the loss of the denoising in the current mask
                self.val_loss_list_N2F.append((torch.sum((outputstest[0]-img_test[0])**2*self.x)/torch.sum(self.x)).cpu())
                self.bg_loss_list.append((torch.sum((img_test[0] - torch.sum(img_test[0]*(1-self.x)))**2*(1-self.x))/torch.sum(1-self.x)).cpu())
                mean = (torch.sum((outputstest[0]*self.x))/torch.sum(self.x)).cpu()
                var = (torch.sum((mean-outputstest[0])**2*self.x)/torch.sum(self.x)).cpu()
                self.variance.append(var)
                cleaned = outputstest[0,0,:,:].cpu().detach().numpy()
                noisy = img_test.cpu().detach().numpy()
                if self.iteration_index>-1:
                    ps = -np.sum((noisy-cleaned)**2*np.asarray(torch.round(self.x.cpu())))/np.sum(np.asarray(torch.round(self.x.cpu())))
                else:
                    ps = -np.mean((noisy-cleaned)**2)
                last10psnr.pop(0)
                last10psnr.append(ps)
                if ps > maxpsnr:
                    maxpsnr = ps
                    outclean = cleaned*maxer+minner
                    timesince = 0
                else:
                    timesince+=1.0
                    # try: 
                    #     running_loss = torch.mean(torch.stack(self.val_loss_list_N2F[-999:-1]))
                    # except:
                    #     running_loss = 1e10
                    # if self.val_loss_list_N2F[-1] > running_loss:
                    #     for g in self.optimizer.param_groups:
                    #         g['lr'] *= 1
        #print('new learning rate is ', g['lr'])
            if timesince > 10:
                H = np.mean(last10, axis=0)
                self.f1 = torch.from_numpy(H).to(self.device)
                self.f1 = self.f1.unsqueeze(0)
                self.en.append(self.compute_energy())
        print('I did ', timesince, ' denoising iterations')
        
        for g in self.optimizer.param_groups:
          g['lr'] *= 0.5
        #print(self.learning_rate)
        plt.plot(self.val_loss_list_N2F_firstmask, label="energy_denoising_firstmask")
        plt.plot(self.val_loss_list_N2F_currentmask, label="energy_denoising_currentmask")

        plt.legend()
        plt.show()

        self.Dice.append(self.Dice[-1])


             
