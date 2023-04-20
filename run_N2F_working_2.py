# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 11:52:26 2023

@author: johan
"""




import os
import numpy as np
from math import sqrt
from Functions_pytorch import *
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from skimage.color import rgb2gray
import scipy
from skimage.transform import resize
from skimage.color import rgb2gray
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model_N2F_working import *
from utils import *
from torch.optim import Adam
import time
import argparse
VERBOSE = 1
# ----

parser = argparse.ArgumentParser(description='Arguments for segmentation network.')
parser.add_argument('--output_directory', type=str, 
                    help='directory for outputs', default="C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data")
parser.add_argument('--input_directory', type=str, 
                    help='directory for input files', default = "C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data")
parser.add_argument('--learning_rate', type=float, 
                    help='learning rate', default=0.00001)
parser.add_argument('--method', type = str, help="joint/sequential or only Chan Vese cv", default = "joint")
parser.add_argument('--lam', type = float, help = "regularization parameter of CV", default = 0.0000001)
parser.add_argument('--ratio', type = int, help = "What is the ratio of masked pixels in N2v", default = 0.3)
parser.add_argument('--experiment', type = str, help = "What hyperparameter are we looking at here? Assigns the folder we want to produce with Lambda, if we make lambda tests for example", default = "/Lambda")
parser.add_argument('--patient', type = int, help = "Which patient index do we use", default = 0)
parser.add_argument('--dataset', type = str, help = "Which dataset", default = "DSB2018_n20")
parser.add_argument('--fid', type = float, help = "do we add fidelity term?", default = 0.0)


device = 'cuda:0'

torch.manual_seed(0)

args = parser.parse_args()
os.chdir("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation")


args.output_directory = args.output_directory+"/" + args.dataset + "/patient_"+ str(args.patient).zfill(2) +  args.experiment
#define directory where you want to have your outputs saved
name = "/S2S_Method_"+ args.method + "_Lambda_" + str(args.lam) + "_ratio_"+ str(args.ratio) +'_lr_'+str(args.learning_rate)
path = args.output_directory+  name
args.lam=0.01
print(path)

def create_dir(dir):
  if not os.path.exists(dir):
    os.makedirs(dir)
    print("Created Directory : ", dir)
  else:
    print("Directory already existed : ", dir)
  return dir
args.patient= 3

create_dir(path)

data = np.load("C:/Users/johan/Desktop/hörnchen/joint_denoising_segmentation/data/"+args.dataset+"/train/train_data.npz")
#
f = torch.tensor(data["X_train"][args.patient:args.patient+1]).to(device)
gt = torch.tensor(data["Y_train"][args.patient:args.patient+1]).to(device)

f_denoising = torch.clone(f)
args.fid = 0.001
mynet = Denseg_S2S(learning_rate = args.learning_rate, lam = args.lam, fid = args.fid)
mynet.initialize(f)
f=mynet.normalize(f)
mynet = Denseg_S2S(learning_rate = args.learning_rate, lam = args.lam, fid = args.fid)
mynet.initialize(f)
f=mynet.normalize(f)
n_it =10
args.method = "joint"
cols = []
first_loss = []
curr_loss = []
median = torch.median(f)
lastepoch = False
if args.method == "joint" or args.method == "cv":
    for i in range(n_it):
      if args.method == "joint":
        mynet.iteration_index = i
        if i <1:
           mynet.x = (mynet.f>0.6).float()
           mynet.first_mask = mynet.x

        else:
            previous_mask = torch.round(mynet.x)
            mynet.first_mask = mynet.x

            mynet.initialize(f)
            mynet.segmentation_step2denoisers_acc_bg_constant(mynet.f, 10000, gt)

            if i > 1:
                #stopping criterion: if the mean value of the image inside the part of the mask that is added by the last iteration, is closer to the bg mean, stop
                mean_inside_mask = torch.sum(torch.round(mynet.x)*f)/torch.sum(torch.round(mynet.x)) 
                mean_inside_difference_ring = torch.sum((torch.round(mynet.x)-previous_mask)*f)/torch.sum((torch.round(mynet.x)-previous_mask))
                mean_in_bg = torch.sum((1-torch.round(mynet.x))*f)/torch.sum(1-torch.round(mynet.x))
                difference_1 = (mean_inside_mask-mean_inside_difference_ring)**2
                difference_2 = (mean_in_bg - mean_inside_difference_ring)**2
                print(difference_1)
                print(difference_2)
                if difference_1 > difference_2 and i >6:
                    print("done")
                    break
            if i == 1:
                mynet.first_mask = mynet.x




            plt.plot(mynet.fidelity_fg, label = "foreground_loss")
            plt.plot(mynet.fidelity_bg[:], label = "background_loss")
            plt.plot(mynet.tv, label = "TV")
            plt.plot(mynet.fidelity_fg_d_bg, label = "fg denoiser on bg")
            plt.plot(mynet.fidelity_bg_d_fg, label = "bg denoiser on fg")
            plt.plot(mynet.difference, label = "fg loss-bg loss on whole image")

            plt.legend()
            plt.show()
            
            plt.plot(mynet.en,label = 'total energy')
            plt.plot(mynet.Dice,label = 'Dice')
            plt.plot("Energy")
            plt.legend()
            plt.show()

  


        if i>-1:
            mynet.denoising_step_r2()
            mynet.N2Fstep()
            #den
            first_loss.append(mynet.previous_loss)
            curr_loss.append(mynet.current_loss)
            try:
                if curr_loss[-1] > curr_loss[-2]  and i>1:
                    print("done")
                    lastepoch = True


                #i = n_it
            except: #
                print('Ich bin Jake')
            


        if i%1 ==0:
            plt.subplot(3,2,1)
            plt.imshow(mynet.x[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,2)
            plt.imshow(f[0].cpu())
            plt.colorbar()
            plt.subplot(3,2,3)
            plt.imshow(mynet.f1[0].cpu(),cmap ='inferno')
            plt.colorbar()
            plt.subplot(3,2,4)
            plt.imshow((mynet.x[0]>0.5).cpu(),cmap ='inferno')
            plt.colorbar()    
            plt.subplot(3,2,5)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.subplot(3,2,6)
            plt.imshow(torch.abs(mynet.f1[0].cpu()-mynet.f[0].cpu()),cmap ='inferno')
            plt.colorbar()   
            plt.show()


            plt.plot(mynet.val_loss_list_N2F, label = "mean denoising performance in current mask")
            plt.title("Denoiser on current mask")
            plt.legend()
            plt.show()

            plt.plot(mynet.variance, label = "variance in mask")
            plt.title("Variance")
            plt.legend()
            plt.show()


