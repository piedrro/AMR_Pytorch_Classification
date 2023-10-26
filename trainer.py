# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:23:14 2022

@author: turnerp
"""
import traceback

import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from skimage import exposure
from datetime import datetime
import os
import pathlib
import itertools
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
# from visalize import generate_plots
import matplotlib.pyplot as plt
import shap
import copy
import warnings
warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.")

class Trainer:

    def __init__(self,
                 model: torch.nn.Module = None,
                 pretrained_model=None,
                 device: torch.device = None,
                 criterion: torch.nn.Module = None,
                 antibiotic_list = [],
                 channel_list = [],
                 optimizer: torch.optim.Optimizer = None,
                 trainloader: torch.utils.data.Dataset = None,
                 valoader: torch.utils.data.Dataset = None,
                 batch_size: int = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 tensorboard=bool,
                 epochs: int = 100,
                 kfolds: int = 0,
                 fold: int = 0,
                 model_folder_name = '',
                 model_path = None,
                 save_dir = '',
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 ):

        self.model = model
        self.pretrained_model = pretrained_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.trainloader = trainloader
        self.valoader = valoader
        self.batch_size = batch_size
        self.device = device
        self.epochs = epochs
        self.model_folder_name = model_folder_name
        self.save_dir = save_dir
        self.model_path = model_path
        self.epoch = 0
        self.tensorboard = tensorboard
        self.antibiotic_list = antibiotic_list
        self.antibiotic = antibiotic_list[-1]
        self.channel_list = channel_list
        self.num_train_images = len(self.trainloader)*self.batch_size
        self.num_validation_images = len(self.valoader)*self.batch_size
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.learning_rate = []
        self.kfolds = kfolds
        self.fold = fold
        self.timestamp = timestamp
        
        condition = [self.antibiotic] + channel_list
        condition = '[' + '-'.join(condition) + ']'

        if os.path.exists(save_dir):
            
            parts = (save_dir,"models",model_folder_name+"_"+self.timestamp,condition,"fold"+str(self.fold))
            model_dir = pathlib.Path('').joinpath(*parts)

        else:
            
            parts = ("models",model_folder_name+"_"+self.timestamp,condition,"fold"+str(self.fold))
            model_dir = pathlib.Path('').joinpath(*parts)

        # model_dir = os.path.abspath(model_dir)
        
        if self.kfolds > 0:
            self.model_path = str(pathlib.Path('').joinpath(*model_dir.parts,"AMRClassification_" + condition + "_" + "fold" + str(self.fold) + "_" + self.timestamp))
                        
        else:
            
            model_dir = pathlib.Path(model_dir)
            self.model_path = str(pathlib.Path('').joinpath(*model_dir.parts[:-1],"AMRClassification_" + condition + "_" + self.timestamp))
                                              

        if not os.path.exists(model_dir):
            print(model_dir)
            os.makedirs(model_dir)

        if pretrained_model:
            if os.path.isfile(pretrained_model):
                model_weights = torch.load(os.path.abspath(pretrained_model))['model_state_dict']
                model.load_state_dict(model_weights)
                
        if tensorboard:
            self.writer = SummaryWriter(log_dir= "runs/" + condition + "_" + timestamp)
            
    def correct_predictions(self, label, pred_label):

        if len(label.shape) > 1:
            correct = (label.data.argmax(dim=1) == pred_label.data.argmax(dim=1)).float().sum().cpu()
        else:
            correct = (label.data == pred_label.data).float().sum().cpu()

        accuracy = correct / label.shape[0]

        return accuracy.numpy()

    def train(self):

        progressbar = tqdm.tqdm(range(self.epochs), 'Progress', total=self.epochs, position=0, leave=False)

        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self.train_step()

            """Validation block"""
            if self.valoader is not None:
                self.val_step()

            """update tensorboard"""
            if self.writer:
                self.writer.add_scalar("Loss/train", self.training_loss[-1], self.epoch)
                self.writer.add_scalar("Loss/validation", self.validation_loss[-1], self.epoch)
                self.writer.add_scalar("Accuracy/train", self.training_accuracy[-1], self.epoch)
                self.writer.add_scalar("Accuracy/validation", self.validation_accuracy[-1], self.epoch)

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()  # learning rate scheduler step

            if self.validation_accuracy[-1] == np.max(self.validation_accuracy):
                
                self.model.eval()
                
                torch.save({'epoch': self.epoch,
                            'num_epochs': self.epochs,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'lr_scheduler': self.lr_scheduler,
                            'training_loss': self.training_loss,
                            'validation_loss': self.validation_loss,
                            'training_accuracy': self.training_accuracy,
                            'validation_accuracy': self.validation_accuracy,
                            'antibiotic': self.antibiotic,
                            'channel_list': self.channel_list,
                            'num_train_images': self.num_train_images,
                            'num_validation_images': self.num_validation_images,
                            'KFOLDS':self.kfolds}, self.model_path)
                
                # model_state_dict = torch.load(self.model_path)['model_state_dict']
                # self.model.load_state_dict(model_state_dict)

            progressbar.set_description(
                f'(Training Loss {self.training_loss[-1]:.5f}, Validation Loss {self.validation_loss[-1]:.5f})')  # update progressbar

        return self.model_path

    def train_step(self):
        
        self.model.train()
        train_losses = []  # accumulate the losses here
        train_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.trainloader), 'Training', total=len(self.trainloader), position=1,
                                leave=False)

        for i, (images, labels) in batch_iter:
            images, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)

            # checks if any images contains a NaN
            if not torch.isnan(images).any():

                self.optimizer.zero_grad()  # zerograd the parameters
                pred_label = self.model(images)  # one forward pass

                loss = self.criterion(pred_label, labels)
                train_losses.append(loss.item())

                accuracy = self.correct_predictions(pred_label, labels)
                train_accuracies.append(accuracy)

                loss.backward()  # one backward pass
                self.optimizer.step()  # update the parameters

                current_lr = self.optimizer.param_groups[0]['lr']

                batch_iter.set_description(
                    f'Training: (loss {np.mean(train_losses):.2f}, Acc {np.mean(train_accuracies):.2f} LR {current_lr})')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.training_accuracy.append(np.mean(train_accuracies))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def val_step(self):

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.valoader), 'Validation', total=len(self.valoader), position=1,
                                leave=False)

        for i, (images, labels) in batch_iter:
            images, labels = images.to(self.device), labels.to(self.device)  # send to device (GPU or CPU)

            #checks if any images contains a NaN
            if not torch.isnan(images).any():

                with torch.no_grad():
                    pred_label = self.model(images)

                    loss = self.criterion(pred_label, labels)
                    valid_losses.append(loss.item())

                    accuracy = self.correct_predictions(pred_label, labels)
                    valid_accuracies.append(accuracy)

                    current_lr = self.optimizer.param_groups[0]['lr']

                    batch_iter.set_description(
                        f'Validation: (loss {np.mean(valid_losses):.2f}, Acc {np.mean(valid_accuracies):.2f} LR {current_lr})')  # update progressbar

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(np.mean(valid_accuracies))

        batch_iter.close()

def normalize99(X):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
    X = X.copy()
    v_min, v_max = np.percentile(X[X != 0], (1, 99))
    X = exposure.rescale_intensity(X, in_range=(v_min, v_max))
    return X


def rescale01(x):
    """ normalize image from 0 to 1 """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x



