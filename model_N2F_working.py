# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:53:00 2023

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
class Denseg_S2S:
    def __init__(
        self,
        learning_rate: float = 1e-3,
        lam: float = 0.01,
        ratio: float = 0.7,
        device: str = 'cuda:0',
        fid: float = 0.1,
        verbose = False,
        
    ):
        self.learning_rate = learning_rate
        self.lam = lam
        self.lam2 = 0.1
        self.ratio = ratio

        self.sigma_fid = 1.0/8
        self.sigma_tv = 1
        #self.tau =  0.95/(4 + 2*5)
        self.tau = 1/100
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
        self.denois_its = 500
        self.loss_s2s=[]
        self.loss_list_N2F=[]
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.energy_denoising = []
        self.val_loss_list_N2F = []
        self.bg_loss_list = []
        self.number_its_N2F=1000
        self.fidelity_fg_d_bg =[]
        self.fidelity_bg_d_fg = []
        self.val_loss_list_N2F_firstmask = []
        self.old_x = 0
        self.fid1 = 0
        self.fid2= 0
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
        f = torch.tensor(f).float()
        f = (f-torch.min(f))/(torch.max(f)-torch.min(f))
        return f
    def standardise(self,f):
        f = torch.tensor(f).unsqueeze(3).float()
        f = (f - torch.mean(f))/torch.std(f)
        return f
        
    def initialize(self,f):
        #prepare input for denoiser
        f_train = torch.tensor(f).unsqueeze(3).float()
  #      f_train = (f_train - torch.mean(f_train))/torch.std(f_train)
        dataset = TensorDataset((f_train[:, :, :]), (f_train[:, :, :]))
        self.train_loader = DataLoader(dataset, batch_size=1, pin_memory=False)
        
        f_val = torch.clone(f_train)
        dataset_val = TensorDataset(f_val, f_val)
        self.val_loader = DataLoader(dataset_val, batch_size=1, pin_memory=False)
        self.f_std = torch.clone(f_train)
        #prepare input for segmentation
        f = self.normalize(f)
        #f = torch.rand_like(f)
        self.p = gradient(f)
        self.q = f
        self.r = f
        self.x_tilde = f
        self.x = f
        self.f = torch.clone(f)


##################### CV segmentation algorithm bg constant############################
    def segmentation_step2denoisers_acc_bg_constant(self,f, iterations, gt):
        f_orig = torch.clone(f).to(self.device)
        f1 = torch.clone(self.f1)

        # compute difference between noisy input and denoised image
        diff1 = (f_orig-f1).float()
        #compute difference between constant of background and originial noisy image
        diff2 = (f_orig - self.mu_r2)
        energy_beginning = torch.sum((diff1)**2*self.x)+ torch.sum((diff2)**2*(1-self.x)) + self.lam*norm1(gradient(self.x))
        print(energy_beginning)
        self.fid1 = diff1
        self.fid2= diff2
        q1 = torch.ones_like(f)
        r1 = torch.ones_like(f)
        self.q = q1.clone()
        self.r = r1.clone()
        '''-------------now the segmentation process starts-------------------'''
        ''' Update dual variables of image f'''
        for i in range(iterations):
            p1 = proj_l1_grad(self.p + self.sigma_tv*gradient(self.x_tilde), self.lam)  # update of TV
    

            # Fidelity term without norm (change this in funciton.py)
            #self.p = p1.clone()
            self.p1 = p1.clone()
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
                fidelity = torch.sum((diff1)**2*self.x)+ torch.sum((diff2)**2*(1-self.x))
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
                #here, we create the list of energy values during the segmentation step
                self.en.append((torch.sum((self.f1-f)**2*self.x) + torch.sum((f -self.mu_r2)**2*(1-self.x)) + self.lam*norm1(gradient(self.x))).cpu())

                
                gt_bin = torch.clone(gt)
                gt_bin[gt_bin > 1] = 1
                seg = torch.round(self.x)
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
        f = torch.clone(self.f_std[:,:,:,0])
        self.mu_r2 = torch.sum((f*(1-self.x))/torch.sum(1-self.x))
        

    
    def reinitialize_network(self):
        self.net = Net()
        self.net.to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)

    def N2Fstep(self):
        self.previous_loss = torch.clone(self.current_loss)

        if self.f1 == None:
            self.f1 = torch.clone(self.f)
        f = torch.clone(self.f_std[:,:,:,0])
        loss_mask=torch.clone(torch.round(self.x)).detach()
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
        #net = Net()
        #net.to(self.device)
        #criterion = torch.sum((output-y)**2*(1-mask)*loss_mask)/torch.sum((1-mask)*loss_mask)

        #criterion = nn.MSELoss()
        #optimizer = optim.Adam(net.parameters(), lr=self.learning_rate)
         
         
        
        img_test = torch.from_numpy(img_test)
        img_test = torch.unsqueeze(img_test,0)
        img_test = torch.unsqueeze(img_test,0)
        img_test = img_test.to(self.device)
        
        running_loss1=0.0
        running_loss2=0.0
        maxpsnr = -np.inf
        timesince = 0
        last10 = [0]*105
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
            loss = loss1
            running_loss1+=loss1.item()
            self.loss_list_N2F.append(loss.detach().cpu())
            loss.backward()
            self.optimizer.step()
             
             
            running_loss1=0.0

            with torch.no_grad():
                last10.pop(0)
                last10.append(cleaned*maxer+minner)
                outputstest = self.net(img_test).detach()

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
  
        print('I did ', timesince, ' denoising iterations')
        H = np.mean(last10, axis=0)

        for g in self.optimizer.param_groups:
          g['lr'] *= 0.5
        self.f1 = torch.from_numpy(H).to(self.device)
        #print(self.learning_rate)
        plt.plot(self.val_loss_list_N2F_firstmask, label="energy_denoising_firstmask")
        plt.plot(self.val_loss_list_N2F_currentmask, label="energy_denoising_currentmask")

        plt.legend()
        plt.show()
        self.f1 = self.f1.unsqueeze(0)
        self.en.append((torch.sum((self.f1-f)**2*self.x) + torch.sum((f -self.mu_r2)**2*(1-self.x)) + self.lam*norm1(gradient(self.x))).cpu())
        self.Dice.append(self.Dice[-1])

        #self.number_its_N2F = 0#self.number_its_N2F*0.9

             
