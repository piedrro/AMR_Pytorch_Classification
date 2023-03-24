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
import optuna
import io
from dataloader import load_dataset
from torch.utils import data
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import cv2
import tifffile
from sklearn.metrics import balanced_accuracy_score
from visualise import generate_plots, process_image, get_image_predictions, normalize99,rescale01

class Trainer:

    def __init__(self,
                 model: torch.nn.Module = None,
                 num_classes: int = None,
                 augmentation: bool = False,
                 pretrained_model=None,
                 device: torch.device = None,
                 antibiotic_list = [],
                 channel_list = [],
                 train_data = None,
                 val_data = None,
                 test_data = None,
                 batch_size: int = None,
                 tensorboard=bool,
                 epochs: int = 100,
                 kfolds: int = 0,
                 fold: int = 0,
                 model_folder_name = '',
                 model_path = None,
                 save_dir = '',
                 timestamp = datetime.now().strftime("%y%m%d_%H%M"),
                 learning_rate = 0.001,
                 ):

        self.model = model
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.pretrained_model = pretrained_model
        self.test_data = test_data
        self.train_data = train_data
        self.val_data = val_data
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
        self.num_train_images = len(train_data)
        self.num_validation_images = len(val_data)
        self.training_loss = []
        self.training_accuracy = []
        self.validation_loss = []
        self.validation_accuracy = []
        self.kfolds = kfolds
        self.fold = fold
        self.timestamp = timestamp
        self.hyperparameters_tuned = False
        self.hyperparameter_study = None

        self.training_dataset = load_dataset(images=train_data["images"], labels=train_data["labels"], num_classes=self.num_classes, augment=self.augmentation)
        self.validation_dataset = load_dataset(images=val_data["images"], labels=val_data["labels"], num_classes=self.num_classes, augment=False)
        self.test_dataset = load_dataset(images=test_data["images"], labels=test_data["labels"], num_classes=self.num_classes, augment=False)

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate_values = []

        self.initialise_model_paths()

        self.bar_format = '{l_bar}{bar:2}{r_bar}{bar:-10b} [{remaining}]'


    def initialise_model_paths(self):

        condition = [self.antibiotic] + self.channel_list
        condition = '[' + '-'.join(condition) + ']'

        if os.path.exists(self.save_dir):
            parts = (self.save_dir, "models", self.model_folder_name + "_" + self.timestamp, condition, "fold" + str(self.fold))
            model_dir = pathlib.Path('').joinpath(*parts)
        else:
            parts = ("models", self.model_folder_name + "_" + self.timestamp, condition, "fold" + str(self.fold))
            model_dir = pathlib.Path('').joinpath(*parts)

        if self.kfolds > 0:
            self.model_dir = str(model_dir)
            self.model_path = str(pathlib.Path('').joinpath(*self.model_dir.parts, "AMRClassification_" + condition + "_" + "fold" + str(self.fold) + "_" + self.timestamp))
        else:
            self.model_dir = pathlib.Path(model_dir)
            self.model_path = str(pathlib.Path('').joinpath(*self.model_dir.parts[:-1], "AMRClassification_" + condition + "_" + self.timestamp))

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if self.pretrained_model:
            if os.path.isfile(self.pretrained_model):
                model_weights = torch.load(os.path.abspath(self.pretrained_model))['model_state_dict']
                self.model.load_state_dict(model_weights)

        if self.tensorboard:
            self.writer = SummaryWriter(log_dir="runs/" + condition + "_" + self.timestamp)


    def correct_predictions(self, label, pred_label):

        if len(label.shape) > 1:
            correct = (label.data.argmax(dim=1) == pred_label.data.argmax(dim=1)).float().sum().cpu()
        else:
            correct = (label.data == pred_label.data).float().sum().cpu()

        accuracy = correct / label.shape[0]

        return accuracy.numpy()

    def visualise_augmentations(self,  n_repeats = 20):

        dataset = load_dataset(images=[self.train_data["images"][1]]*n_repeats, labels=[self.train_data["labels"][1]]*n_repeats, num_classes=self.num_classes, augment=True)
        dataloader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)

        augmented_images = []

        for images, _ in dataloader:

            img = images[0].numpy()

            img = process_image(img)

            augmented_images.append(img)

        fig, ax = plt.subplots(4, 5, figsize=(12, 10))
        for i in range(4):
            for j in range(5):
                ax[i, j].imshow(augmented_images[i*4+j], cmap='gray')
                ax[i, j].axis('off')

        fig.suptitle('Example Augmentations', fontsize=16)
        fig.tight_layout()
        plt.show()


    def optuna_objective(self, trial):

        batch_size = trial.suggest_int("batch_size", 10, 200, log=True)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-1)

        tune_trainloader = data.DataLoader(dataset=self.tune_train_dataset, batch_size=batch_size, shuffle=False)
        tune_valoader = data.DataLoader(dataset=self.tune_val_dataset, batch_size=batch_size, shuffle=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        model = copy.deepcopy(self.model)
        model.to(self.device)

        running_loss = 0.0

        for images, labels in tune_trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            if not torch.isnan(images).any():
                self.optimizer.zero_grad()
                pred_label = model(images)
                loss = self.criterion(pred_label, labels)
                loss.backward()
                self.optimizer.step()

        for images, labels in tune_valoader:
            images, labels = images.to(self.device), labels.to(self.device)
            if not torch.isnan(images).any():
                pred_label = model(images)
                loss = self.criterion(pred_label, labels)
                running_loss += loss.item()

        return running_loss/len(tune_valoader)


    def load_tune_dataset(self, num_images=100, num_epochs = 10):

        tune_images = []
        tune_labels = []

        tune_train_dataset = load_dataset(images=self.train_data["images"][:num_images], labels=self.train_data["labels"][:num_images], num_classes=self.num_classes, augment=True)
        tuneloader = data.DataLoader(dataset=tune_train_dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(num_epochs):
            for images, labels in tuneloader:
                for image in images:
                    image = image.numpy()
                    tune_images.append(image)
                for label in labels:
                    label = int(label.argmax(dim=0).numpy())
                    tune_labels.append(label)

        print(f"Loaded {len(tune_images)} images for hyperparameter tuning.")

        self.tune_train_dataset = load_dataset(images=tune_images, labels=tune_labels, num_classes=self.num_classes, augment=False)
        self.tune_val_dataset = load_dataset(images=self.val_data["images"][:num_images], labels=self.val_data["labels"][:num_images], num_classes=self.num_classes, augment=False)


    def tune_hyperparameters(self, num_trials=5, num_images = 500, num_epochs = 4):

        self.load_tune_dataset(num_images = num_images, num_epochs = num_epochs)

        self.num_tune_images = num_images
        self.num_tune_epochs = num_epochs

        study = optuna.create_study(direction='minimize')
        study.optimize(self.optuna_objective, n_trials=num_trials)

        print("Best trial:")
        trial = study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        self.batch_size = int(trial.params["batch_size"])
        self.learning_rate = float(trial.params["learning_rate"])
        self.hyperparameter_study = study

        optuna.visualization.plot_optimization_history(study).write_image(self.model_dir.joinpath(f"optuna_plot_optimization_history.png"))
        optuna.visualization.plot_slice(study).write_image(self.model_dir.joinpath(f"optuna_plot_slice.png"))
        optuna.visualization.plot_parallel_coordinate(study).write_image(self.model_dir.joinpath(f"optuna_plot_parallel_coordinate.png"))
        optuna.visualization.plot_contour(study).write_image(self.model_dir.joinpath(f"optuna_plot_contour.png"))
        optuna.visualization.plot_param_importances(study).write_image(self.model_dir.joinpath(f"optuna_plot_param_importances.png"))

        from PIL import Image
        img = np.asarray(Image.open(self.model_dir.joinpath(f"optuna_plot_slice.png")))
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        return study

    def train(self):

        self.trainloader = data.DataLoader(dataset=self.training_dataset, batch_size=self.batch_size, shuffle=True)
        self.valoader = data.DataLoader(dataset=self.validation_dataset, batch_size=self.batch_size, shuffle=False)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

        progressbar = tqdm.tqdm(range(self.epochs), 'Progress', total=self.epochs, position=0, leave=True, bar_format=self.bar_format)

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
                            'KFOLDS':self.kfolds,
                            'hyperparameters_tuned':self.hyperparameters_tuned,
                            'hyperparameter_study':self.hyperparameter_study}, self.model_path,)
                
                # model_state_dict = torch.load(self.model_path)['model_state_dict']
                # self.model.load_state_dict(model_state_dict)

            progressbar.set_description(
                f'(Training Loss {self.training_loss[-1]:.5f}, Validation Loss {self.validation_loss[-1]:.5f})')  # update progressbar

        return self.model_path

    def train_step(self):
        
        self.model.train()
        train_losses = []  # accumulate the losses here
        train_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.trainloader), 'Training', total=len(self.trainloader), position=0,leave=False, bar_format=self.bar_format)

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

                batch_iter.set_description(f'Training[{self.epoch}\\{self.epochs}]:(loss {np.mean(train_losses):.3f}, Acc {np.mean(train_accuracies):.2f}')  # update progressbar

        self.training_loss.append(np.mean(train_losses))
        self.training_accuracy.append(np.mean(train_accuracies))
        self.learning_rate_values.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def val_step(self):

        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_accuracies = []

        batch_iter = tqdm.tqdm(enumerate(self.valoader), 'Validation', total=len(self.valoader), position=0,leave=False, bar_format=self.bar_format)

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
                        f'Validation[{self.epoch}\\{self.epochs}]:(loss{np.mean(valid_losses):.3f}, Acc{np.mean(valid_accuracies):.2f}')  # update progressbar

        self.validation_loss.append(np.mean(valid_losses))
        self.validation_accuracy.append(np.mean(valid_accuracies))

        batch_iter.close()

    def evaluate(self, model_path):
        
        model_data = torch.load(model_path)
        self.model.load_state_dict(model_data['model_state_dict'])

        self.model.eval()  # evaluation mode

        test_images = []
        saliency_maps = []
        true_labels = []
        pred_labels = []
        pred_losses = []
        pred_confidences = []

        testloader = torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

        batch_iter = tqdm.tqdm(enumerate(testloader), 'Evaluating', total=len(testloader), position=0, leave=False)

        self.model.to(self.device)

        for i, (image, label) in batch_iter:

            try:

                image, label = image.to(self.device), label.to(self.device)  # send to device (GPU or CPU)

                if not torch.isnan(image).any():

                    image.requires_grad = True

                    pred_label = self.model(image)
                    loss = self.criterion(pred_label, label)
                    outputs = F.softmax(pred_label, dim=1)
                    _, predicted = torch.max(outputs.data, 1)

                    # Backward pass to get gradients
                    one_hot = torch.zeros(1, outputs.size()[-1])
                    one_hot[0, predicted] = 1
                    outputs.backward(gradient=one_hot.to(self.device))

                    # Get the gradients of the input image
                    gradients = image.grad.data.squeeze().cpu().numpy()
                    saliency_map = np.max(np.abs(gradients), axis=0)

                    plot_image = image.squeeze().cpu().detach().numpy()

                    saliency_map = rescale01(saliency_map)
                    saliency_map = saliency_map * 255
                    saliency_map = saliency_map.astype(np.uint8)

                    saliency_map = cv2.cvtColor(saliency_map, cv2.COLOR_GRAY2RGB)
                    saliency_map = rescale01(saliency_map)

                    saliency_maps.append(saliency_map)

                    plot_image = process_image(plot_image)
                    test_images.append(plot_image)

                    pred_confidences.append(torch.nn.functional.softmax(pred_label, dim=1).tolist())
                    pred_labels.append(pred_label.data.cpu().argmax().numpy().tolist())
                    true_labels.append(label.data.cpu().argmax().numpy().tolist())
                    pred_losses.append(loss.item())

            except:
                # print(traceback.format_exc())
                pass

        accuracy = self.correct_predictions(torch.tensor(true_labels), torch.tensor(pred_labels))

        test_predictions = get_image_predictions(test_images,
                                                 saliency_maps,
                                                 true_labels,pred_labels,
                                                 pred_confidences,
                                                 self.antibiotic_list)

        cm = confusion_matrix(true_labels, pred_labels, normalize='pred')

        model_data["confusion_matrix"] = cm
        model_data["test_labels"] = true_labels
        model_data["pred_labels"] = pred_labels
        model_data["test_accuracy"] = accuracy
        model_data["num_test_images"] = len(test_images)
        model_data["test_predictions"] = test_predictions
        model_data["test_balanced_accuracy"] = balanced_accuracy_score(true_labels, pred_labels)

        generate_plots(model_data, self.model_path)
        torch.save(model_data, self.model_path)

        return




