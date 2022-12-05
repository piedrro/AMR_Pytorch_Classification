from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit

from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models

import pickle
import matplotlib.pyplot as plt
import os
import numpy as np
from datetime import datetime
import pandas as pd
import itertools
from itertools import compress, product
from skimage import exposure
import cv2
import json
import pickle
import random
# from visalize import generate_plots
from trainer import Trainer
from file_io import get_metadata, get_cell_images, cache_data, get_training_data
from dataloader import load_dataset
from model import timm_model




image_size = (64,64)
resize = False
antibiotic_list = ["Untreated", "Chloramphenicol"]
microscope_list = ["BIO-NIM"]
channel_list = ["405","532"]
cell_list = ["single"]
train_metadata = {"segmentation_curated":True}
test_metadata = {}

model_backbone = 'densenet121'
ratio_train = 0.9
val_test_split = 0.5
BATCH_SIZE = 10
LEARNING_RATE = 0.001
EPOCHS = 10
AUGMENT = True

AKSEG_DIRECTORY = r"\\physics.ox.ac.uk\dfs\DAQ\CondensedMatterGroups\AKGroup\Piers\AKSEG"
USER_INITIAL = "AF"
SAVE_DIR = r"C:\Users\turnerp\PycharmProjects\AMR_GramStain\models"
MODEL_FOLDER_NAME = "AntibioticClassification"


# device
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')
    
    
akseg_metadata = get_metadata(AKSEG_DIRECTORY,
                              USER_INITIAL,
                              channel_list,
                              antibiotic_list,
                              microscope_list,
                              train_metadata,
                              test_metadata)







if __name__ == '__main__':
    
    # cached_data = cache_data(akseg_metadata,
    #                           image_size,
    #                           antibiotic_list,
    #                           channel_list,
    #                           cell_list,
    #                           import_limit = 10,
    #                           mask_background=True,
    #                           resize=resize)
    
    # with open('cacheddata.pickle', 'wb') as handle:
    #     pickle.dump(cached_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cacheddata.pickle', 'rb') as handle:
        cached_data = pickle.load(handle)
        
    num_classes = len(np.unique(cached_data["labels"]))
        
    train_data, val_data, test_data  = get_training_data(cached_data,
                                                          shuffle=True,
                                                          ratio_train = 0.9,
                                                          val_test_split=0.5,
                                                          label_limit = 'None')

    training_dataset = load_dataset(images = train_data["images"],
                                    labels = train_data["labels"],
                                    num_classes = num_classes,
                                    augment=AUGMENT)
    
    validation_dataset = load_dataset(images = val_data["images"],
                                      labels = train_data["labels"],
                                      num_classes = num_classes,
                                      augment=False)
    
    test_dataset = load_dataset(images = test_data["images"],
                                labels = train_data["labels"],
                                num_classes = num_classes,
                                augment=False)

    trainloader = data.DataLoader(dataset=training_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    
    valoader = data.DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    
    testloader = data.DataLoader(dataset=test_dataset,
                                  batch_size=1,
                                  shuffle=False)

    # images, labels = next(iter(trainloader))
    
    # for img in images:
        
    #     img = np.concatenate(np.split(img.numpy(),3,axis=0),axis=2)[0]
        
    #     plt.imshow(img)
    #     plt.show()

    


    # model = timm_model(num_classes, model_backbone, pretrained=True).to(device) 
    model = models.densenet121(pretrained=False, num_classes=len(antibiotic_list)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    

    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      trainloader=trainloader,
                      valoader=valoader,
                      lr_scheduler=scheduler,
                      tensorboard=True,
                      antibiotic_list = antibiotic_list,
                      channel_list = channel_list,
                      epochs=EPOCHS,
                      batch_size = BATCH_SIZE,
                      model_folder_name = MODEL_FOLDER_NAME)
    
    # model_path = trainer.train()
    
    
    
    
    
    
    
    
    
    
    model_path = r"C:\Users\turnerp\PycharmProjects\AMR_Pytorch_Classification\models\AntibioticClassification_221202_1548\[Chloramphenicol-405-532]\AMRClassification_[Chloramphenicol-405-532]_221202_1548"

    model_data = torch.load(model_path)
    model = model.load_state_dict(model_data['model_state_dict'])
    
    
    # from timm_vis.methods import *
    # import tifffile
    
    # model = timm_model(num_classes, "densenet121", pretrained=True).to(device) 
    
    # model.eval()
    # model.to(device)
    
    # image, label = next(iter(trainloader))
    
    # for img in image:
        
    #     img = img.unsqueeze(dim=0).to(device)
    

    #     img.requires_grad = True
    #     out = model(img)
    #     max_out = out.max()
    #     max_out.backward()    
    #     saliency, _ = torch.max(img.grad.data.abs(), dim = 1)
    #     saliency = saliency.squeeze(0)
    #     saliency_img = saliency.detach().cpu().numpy()

    #     plt.imshow(saliency_img)
    #     plt.show()

    
















