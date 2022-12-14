# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 14:23:14 2022

@author: turnerp
"""

import numpy as np
import torch
import tqdm
import torch.nn.functional as F
from skimage import exposure
from datetime import datetime
import os
import pathlib
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
# from visalize import generate_plots
import matplotlib.pyplot as plt
import shap
import copy
import warnings
warnings.filterwarnings("ignore", message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated and will be removed in future versions. This hook will be missing some grad_input. Please use register_full_backward_hook to get the documented behavior.")




def get_image_predictions(images,saliency,test_labels,pred_labels,pred_confidences,antibiotic_list):

    images_TP = []
    images_TN = []
    images_FP = []
    images_FN = []
    
    saliency_TP = []
    saliency_TN = []
    saliency_FP = []
    saliency_FN = []
    
    label_TP = None
    label_TN = None
    label_FP = None
    label_FN = None
    
    predicted_label_TP = None
    predicted_label_TN = None
    predicted_label_FP = None
    predicted_label_FN = None
    
    confidence_TN = []
    confidence_TP = []
    confidence_FN = []
    confidence_FP = []
    
    for i in range(len(images)):
        
        test_label = test_labels[i]
        pred_label = pred_labels[i]
        
        if test_label == 1 and pred_label == 1:
            images_TP.append(images[i])
            saliency_TP.append(saliency[i])
            label_TP = antibiotic_list[test_label]
            predicted_label_TP = antibiotic_list[pred_label]
            confidence_TP.append(pred_confidences[i][0][pred_label])
            
        if test_label == 0 and pred_label == 0:
            images_TN.append(images[i])
            saliency_TN.append(saliency[i])
            label_TN = antibiotic_list[test_label]
            predicted_label_TN = antibiotic_list[pred_label]
            confidence_TN.append(pred_confidences[i][0][pred_label])
                
        if test_label == 0 and pred_label == 1:
            images_FP.append(images[i])
            saliency_FP.append(saliency[i])
            label_FP = antibiotic_list[test_label]
            predicted_label_FP = antibiotic_list[pred_label]
            confidence_FP.append(pred_confidences[i][0][pred_label])
            
        if test_label == 1 and pred_label == 0:
            images_FN.append(images[i])
            saliency_FN.append(saliency[i])
            label_FN = antibiotic_list[test_label]
            predicted_label_FN = antibiotic_list[pred_label]
            confidence_FN.append(pred_confidences[i][0][pred_label])
            
    miss_predictions = {}
    
    if len(images_TP) > 0:
        images_TP, saliency_TP, confidence_TP = [list(x) for x in zip(*sorted(zip(images_TP, saliency_TP, confidence_TP), key=lambda x: x[2]))]
    if len(images_TN) > 0:
        images_TN, saliency_TN, confidence_TN = [list(x) for x in zip(*sorted(zip(images_TN, saliency_TN, confidence_TN), key=lambda x: x[2]))]
    if len(images_FP) > 0:
        images_FP, saliency_FP, confidence_FP = [list(x) for x in zip(*sorted(zip(images_FP, saliency_FP, confidence_FP), key=lambda x: x[2]))]
    if len(images_FN):
        images_FN, saliency_FN, confidence_FN = [list(x) for x in zip(*sorted(zip(images_FN, saliency_FN, confidence_FN), key=lambda x: x[2]))]
    
    miss_predictions["True Positives"] = {"images":images_TP,
                                          "saliency_map":saliency_TP,
                                          "true_label": label_TP,
                                          "predicted_label": predicted_label_TP,
                                          "prediction_confidence": confidence_TP}
    
    miss_predictions["True Negatives"] = {"images":images_TN,
                                          "saliency_map":saliency_TN,
                                          "true_label": label_TN,
                                          "predicted_label": predicted_label_TN,
                                          "prediction_confidence": confidence_TN}   
    
    miss_predictions["False Positives"] = {"images":images_FP,
                                           "saliency_map":saliency_FP,
                                           "true_label": label_FP,
                                           "predicted_label": predicted_label_FP,
                                           "prediction_confidence": confidence_FP}
    

    miss_predictions["False Negatives"] = {"images":images_FN,
                                           "saliency_map":saliency_FN,
                                           "true_label": label_FN,
                                           "predicted_label": predicted_label_FN,
                                           "prediction_confidence": confidence_FN}  
    
    return miss_predictions






def generate_shap_image(deep_explainer,test_image):

    shap_values = deep_explainer.shap_values(test_image)
    
    shap_values = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2)[0].sum(-1) for s in shap_values]
    test_image = np.swapaxes(np.swapaxes(test_image.cpu().numpy(), 1, -1), 1, 2)
    
    shap_img = np.zeros(test_image.shape[1:])
    
    for i in range(len(shap_values)):
        
        sv = shap_values[i]
        
        v_min, v_max = np.percentile(sv[sv > 0], (1, 99))
        sv = exposure.rescale_intensity(sv, in_range=(v_min, v_max))
        
        sv = (sv - np.min(sv)) / (np.max(sv) - np.min(sv))
        
        if i == 0:
            index = 2
        if i == 1:
            index = 0
        
        shap_img[:,:,index] = sv
        
    return shap_img


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
            model_dir = save_dir + "/models/" + model_folder_name + "_" + self.timestamp + "/" + condition + "/" + "fold" + str(self.fold) + "/"
        else:
            model_dir = "models/" + model_folder_name + "_" + self.timestamp + "/" + condition + "/" + "fold" + str(self.fold) + "/"
            
        model_dir = os.path.abspath(model_dir)
        
        if self.kfolds > 0:
            self.model_path = model_dir + "AMRClassification_" + condition + "_" + "fold" + str(self.fold) + "_" + self.timestamp
        else:
            model_dir = model_dir.replace(model_dir.split("\\")[-1],"")
            self.model_path = model_dir + "AMRClassification_" + condition + "_" + self.timestamp
        
  
        if not os.path.exists(model_dir):
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

            if self.validation_loss[-1] == np.min(self.validation_loss):
                
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
                f'Training: (loss {np.mean(train_losses):.10f}, Acc {np.mean(train_accuracies):.2f} LR {current_lr})')  # update progressbar

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

            with torch.no_grad():
                pred_label = self.model(images)

                loss = self.criterion(pred_label, labels)
                valid_losses.append(loss.item())

                accuracy = self.correct_predictions(pred_label, labels)
                valid_accuracies.append(accuracy)

                current_lr = self.optimizer.param_groups[0]['lr']

                batch_iter.set_description(
                    f'Validation: (loss {np.mean(valid_losses):.10f}, Acc {np.mean(valid_accuracies):.2f} LR {current_lr})')  # update progressbar

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(np.mean(valid_accuracies))

        batch_iter.close()

    def evaluate(self, model, model_path, train_images, test_images, test_labels, num_classes):
        
        model_data = torch.load(model_path)
        model = model.load_state_dict(model_data['model_state_dict'])

        # self.model.eval()  # evaluation mode

        # saliency_maps = []
        # true_labels = []
        # pred_labels = []
        # pred_losses = []
        # pred_confidences = []
        
        # progressbar = tqdm.tqdm(range(len(test_images)), desc='Evaluating', position=0, leave=False)

        # for i, x, y in zip(progressbar, test_images, test_labels):
        #     # Typecasting
        #     x = torch.from_numpy(x.copy()).float()
        #     y = F.one_hot(torch.tensor(y), num_classes=num_classes).float()

        #     x = torch.unsqueeze(x, 0)
        #     y = torch.unsqueeze(y, 0)

        #     image, label = x.to(self.device), y.to(self.device)
            
        #     with torch.no_grad():
        #         pred_label = self.model(image)  # send through model/network

        #         loss = self.criterion(pred_label, label)

        #         pred_confidences.append(torch.nn.functional.softmax(pred_label, dim=1).tolist())
        #         pred_labels.append(pred_label.data.cpu().argmax().numpy().tolist())
        #         true_labels.append(label.data.cpu().argmax().numpy().tolist())
        #         pred_losses.append(loss.item())
        
        # progressbar = tqdm.tqdm(range(len(test_images)), desc='Generating Saliency Maps', position=0, leave=False)
        
        # train_images = torch.from_numpy(np.stack(train_images[:100])).float().to(self.device)
        # deep_explainer = shap.DeepExplainer(self.model.eval(), train_images)
            
        # for i, x, y in zip(progressbar, test_images, test_labels):
            
        #     x = torch.from_numpy(x.copy()).float()
        #     y = F.one_hot(torch.tensor(y), num_classes=num_classes).float()

        #     x = torch.unsqueeze(x, 0)
        #     y = torch.unsqueeze(y, 0)

        #     image, label = x.to(self.device), y.to(self.device) 
        
        #     shap_img = generate_shap_image(deep_explainer,image)
        #     saliency_maps.append(shap_img)

                
        # accuracy = self.correct_predictions(torch.tensor(true_labels), torch.tensor(pred_labels))   

        # test_predictions = get_image_predictions(test_images,
        #                                          saliency_maps,
        #                                          true_labels,pred_labels,
        #                                          pred_confidences,
        #                                          self.antibiotic_list)
            
        # cm = confusion_matrix(true_labels, pred_labels, normalize='pred')
        
        # model_data["confusion_matrix"] = cm
        # model_data["test_labels"] = true_labels
        # model_data["pred_labels"] = pred_labels
        # model_data["test_accuracy"] = accuracy
        # model_data["num_test_images"] = len(test_images)
        # model_data["test_predictions"] = test_predictions
        
        # torch.save(model_data, self.model_path)
        
        # generate_plots(model_data, self.model_path)
        
        return model_data


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



