import os
import ntpath
import time
import datetime
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .network import *
from .test_dataset import *
from .utils import *


def WGAN_tester(opt):
    results_path = opt.results_path
    # Save the model if pre_train == True
    def load_model_generator(net, epoch, opt):
        model_name = 'deepfillv2_WGAN_G_epoch%d_batchsize%d.pth' % (epoch, 4)
        # [repro] resolve relative to this file instead of the current working directory
        model_name = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_model', model_name)
        pretrained_dict = torch.load(model_name, map_location='cpu')
        generator.load_state_dict(pretrained_dict)

    # ----------------------------------------
    #      Initialize training parameters
    # ----------------------------------------

    # configurations
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Build networks
    generator = create_generator(opt).eval()
    print('-------------------------Loading Pretrained Model-------------------------')
    load_model_generator(generator, opt.epoch, opt)
    print('-------------------------Pretrained Model Loaded-------------------------')

    # To device
    generator = generator.cuda()
    
    # ----------------------------------------
    #       Initialize training dataset
    # ----------------------------------------

    # Define the dataset
    trainset = InpaintDataset(opt)
    print('The overall number of images equals to %d' % len(trainset))

    # Define the dataloader
    dataloader = DataLoader(trainset, batch_size = opt.batch_size, shuffle = False, num_workers = opt.num_workers,
                            pin_memory = True)
    
    # ----------------------------------------
    #            Testing
    # ----------------------------------------
    # Testing loop
    name_list = trainset.imglist  # [repro] same order as the dataset, computed once
    skipped = 0
    for batch_idx, (img, mask) in enumerate(dataloader):
        # [repro] resume: skip images whose inpainted result already exists (same naming as save_sample_png)
        out_name = ntpath.basename(name_list[int(batch_idx)]).replace('.png', '').replace(' ', '_') + '.png'
        if os.path.exists(os.path.join(results_path, out_name)):
            skipped += 1
            continue
        img = img.cuda()
        mask = mask.cuda()

        # Generator output
        with torch.no_grad():
            first_out, second_out = generator(img, mask)

        # forward propagation
        first_out_wholeimg = img * (1 - mask) + first_out * mask        # in range [0, 1]
        second_out_wholeimg = img * (1 - mask) + second_out * mask      # in range [0, 1]

        masked_img = img * (1 - mask) + mask
        mask = torch.cat((mask, mask, mask), 1)
        img_list = [second_out_wholeimg]
        save_sample_png(sample_folder=results_path, sample_name='%d' % (batch_idx + 1), img_list=img_list,
                        name_list=[name_list[int(batch_idx)]], pixel_max_cnt=255)
