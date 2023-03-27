from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedShuffleSplit
from torch.utils import data
from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
# from trainer import Trainer
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
from imgaug import augmenters as iaa
import albumentations as A
import tifffile
import tqdm
from multiprocessing import Pool
import traceback
from functools import partial
import random
import torch.nn.functional as F
import pathlib
from skimage.registration import phase_cross_correlation
import scipy
from visualise import normalize99,rescale01
from stats import get_stats


def align_images(images):

    try:

        shift, error, diffphase = phase_cross_correlation(images[0], images[1], upsample_factor=100)
        images[1] = scipy.ndimage.shift(images[1], shift).astype(np.uint16)

    except:
        pass

    return images


def extract_list(data, mode = "file"):
    
    data = data.strip("[]").replace("'","").split(", ")
    
    return data

def update_akseg_paths(path, AKSEG_DIRECTORY, USER_INITIAL):
    
    try:
    
        path = pathlib.Path(path.replace("\\","/"))
        AKSEG_DIRECTORY = pathlib.Path(AKSEG_DIRECTORY)
            
        
        index = path.parts.index(str(USER_INITIAL))
        
        parts = (*AKSEG_DIRECTORY.parts, "Images", *path.parts[index:])
        path = pathlib.Path('').joinpath(*parts)

    except:
        path = None

    return path


def check_channel_list(dat, target_channel_list):
    
    channel_list = dat["channel_list"]
    file_list = dat["file_list"]
    
    if set(target_channel_list) <= set(channel_list):

        indices = [channel_list.index(value) for value in target_channel_list]
        
        file_list = np.take(file_list,indices).tolist()
        
        dat['file_list'] = file_list
        dat['channel_list'] = target_channel_list
        
    else:
        
        dat["file_list"] = None
        dat["channel_list"] = None
        
        pass
    
    return dat



def get_metadata(AKSEG_DIRECTORY, USER_INITIAL,
                 channel_list, antibiotic_list=[], microscope_list=[],
                 train_metadata = {}, test_metadata={}, limit = "None"):

    path = os.path.join(AKSEG_DIRECTORY,"Images", USER_INITIAL, f"{USER_INITIAL}_file_metadata.txt")
    
    akseg_metadata = pd.read_csv(path, sep = ",", low_memory=False)
    
    akseg_metadata["file_list"] = akseg_metadata["file_list"].apply(lambda data: extract_list(data))
    akseg_metadata["channel_list"] = akseg_metadata["channel_list"].apply(lambda data: extract_list(data, mode = "channel"))
    
    akseg_metadata["image_save_path"] = akseg_metadata["image_save_path"].apply(lambda path: update_akseg_paths(path, AKSEG_DIRECTORY, USER_INITIAL))
    akseg_metadata["mask_save_path"] = akseg_metadata["mask_save_path"].apply(lambda path: update_akseg_paths(path, AKSEG_DIRECTORY, USER_INITIAL))
    akseg_metadata["label_save_path"] = akseg_metadata["label_save_path"].apply(lambda path: update_akseg_paths(path, AKSEG_DIRECTORY, USER_INITIAL))
    
    akseg_metadata = akseg_metadata.drop_duplicates(subset=['akseg_hash'], keep="first")

    if len(channel_list) > 0:
        
        akseg_metadata = akseg_metadata[akseg_metadata["channel"].isin(channel_list)]
        akseg_metadata = akseg_metadata.apply(lambda dat: check_channel_list(dat, channel_list), axis=1)
        akseg_metadata = akseg_metadata[~akseg_metadata["file_list"].isin([None, "None", np.nan])]

    akseg_metadata = akseg_metadata.drop_duplicates(["segmentation_file","folder","content"])

    akseg_metadata = akseg_metadata[akseg_metadata["segmentation_file"] != "missing image channel"]

    akseg_metadata = akseg_metadata[akseg_metadata['channel'].notna()]
    akseg_metadata = akseg_metadata.reset_index(drop=True)

    akseg_metadata["dataset"] = ""

    if len(train_metadata) > 0:

        train = akseg_metadata.copy()

        for key,value in train_metadata.items():

            train = train[train[key] == value]

        train_indices = train.index.values.tolist()

        akseg_metadata.loc[akseg_metadata.index.isin(train_indices), "dataset"] = "train"

    if len(test_metadata) > 0:

        test = akseg_metadata.copy()

        for key,value in test_metadata.items():

            test = test[test[key] == value]

        test_indices = test.index.values.tolist()

        akseg_metadata.loc[akseg_metadata.index.isin(test_indices), "dataset"] = "test"

    akseg_metadata = akseg_metadata[akseg_metadata["dataset"] != ""]

    akseg_metadata.loc[akseg_metadata["antibiotic"].isin(["",None, np.nan,"None"]), ["antibiotic"]] = "Untreated"

    if len(antibiotic_list) > 0:
        akseg_metadata = akseg_metadata[akseg_metadata["antibiotic"].isin(antibiotic_list)]

    if len(microscope_list) > 0:
        akseg_metadata = akseg_metadata[akseg_metadata["microscope"].isin(microscope_list)]

    if limit != 'None':
        akseg_metadata = akseg_metadata.sample(frac=1, random_state=42).iloc[:limit]

    return akseg_metadata

def import_coco_json(json_path, cell_types):
    
    cell_dict = {"single":1, "dividing" : 2, "divided" : 3, "vertical" : 4, "broken" : 5, "edge" : 6}
    
    target_classes = [val for key, val in cell_dict.items() if key in cell_types]
    
    with open(json_path, 'r') as f:
        dat = json.load(f)

    h = dat["images"][0]["height"]
    w = dat["images"][0]["width"]

    mask = np.zeros((h, w), dtype=np.uint16)
    labels = np.zeros((h, w), dtype=np.uint16)

    categories = {}

    for i, cat in enumerate(dat["categories"]):
        
        cat_id = cat["id"]
        cat_name = cat["name"]

        categories[cat_id] = cat_name

    annotations = dat["annotations"]

    for i in range(len(annotations)):
        
        category_id = annotations[i]["category_id"]
        
        if category_id in target_classes:
            
            annot = annotations[i]["segmentation"][0]
            
            cnt = np.array(annot).reshape(-1, 1, 2).astype(np.int32)
    
            cv2.drawContours(mask, [cnt], contourIdx=-1, color=i + 1, thickness=-1)
            cv2.drawContours(labels, [cnt], contourIdx=-1, color=category_id, thickness=-1)

    return mask, labels


def resize_image(image_size, h, w, cell_image_crop, colicoords = False, resize=False):
    
    cell_image_crop = np.moveaxis(cell_image_crop, 0, -1)

    if h < image_size[0] and w < image_size[1]:
        
        seq = iaa.CenterPadToFixedSize(height=image_size[0], width=image_size[1])
 
    else:
        
        if resize == True:
        
            if h > w:
                seq = iaa.Sequential([iaa.Resize({"height": image_size[0], "width": "keep-aspect-ratio"}),
                                      iaa.CenterPadToFixedSize(height=image_size[0],width=image_size[1])])
            else:
                seq = iaa.Sequential([iaa.Resize({"height": "keep-aspect-ratio", "width": image_size[1]}),
                                      iaa.CenterPadToFixedSize(height=image_size[0],width=image_size[1])])
                
        else:
            seq = iaa.Sequential([iaa.CenterPadToFixedSize(height=image_size[0],width=image_size[1]),
                                  iaa.CenterCropToFixedSize(height=image_size[0],width=image_size[1])])

    seq_det = seq.to_deterministic()
    cell_image_crop = seq_det.augment_images([cell_image_crop])[0]
        
        
    cell_image_crop = np.moveaxis(cell_image_crop, -1, 0)
    
    if colicoords:
        if h > w:
            cell_image_crop = np.rot90(cell_image_crop,axes=(1,2))


    return cell_image_crop


def get_crop_range(image, image_size, coords):
    
    y1,y2,x1,x2 = coords
    
    cy = int(np.mean([y2, y1]))
    cx = int(np.mean([x2, x1]))
    
    cy1 = cy - image_size[1]//2
    cy2 = cy + image_size[1]//2
    
    cx1 = cx - image_size[0]//2
    cx2 = cx + image_size[0]//2
    
    if cx1 > 0 and cy1 > 0 and cx2 < image.shape[1] and cy2 < image.shape[0]:
        
        coords = [cy1,cy2,cx1,cx2]
        
    else:
        coords = None
    
    return coords




def get_cell_images(dat, image_size, channel_list, cell_list, antibiotic_list,
                    import_limit, colicoords, mask_background, normalise = True, resize=False, align=True):

    try:

        file_names = dat["file_list"].iloc[0]
        channels = dat["channel_list"].iloc[0]

        file_dir = os.path.dirname(dat["image_save_path"].tolist()[0])

        dat_antibiotic = dat["antibiotic"].tolist()[0]
        dat_dataset = dat["dataset"].tolist()[0]

        #segmentation_file = file_names[channels.index("532")]

        # creates RGB image from image channel(s)

        image_channels = []
        image_data = []

        for i in range(len(file_names)):

            file_name = file_names[i]
            channel = channels[i]

            if channel in channel_list:

                img_path = os.path.abspath(os.path.join(file_dir,file_name))

                json_path = pathlib.Path(img_path.replace(".tif", ".txt"))
                index = json_path.parts.index("images")
                parts = (*json_path.parts[:index], "json", *json_path.parts[index+1:])
                json_path = pathlib.Path('').joinpath(*parts)

                img = tifffile.imread(img_path)
                image_data.append(img)
                image_channels.append(channel)

                mask, labels = import_coco_json(json_path, cell_list)

        if align:
            image_data = align_images(image_data)

        if len(image_channels) <= 3:
            rgb = np.zeros((3, image_data[0].shape[0],image_data[0].shape[1]))
        else:
            rgb = np.zeros((len(image_channels), image_data[0].shape[0],image_data[0].shape[1]))

        for i in range(len(image_data)):

            channel = image_channels[i]
            channel_index = list(channel_list).index(channel)

            rgb[channel_index,:,:] = image_data[i]

        # crops images and gets labels

        cell_dataset = []
        cell_images = []
        cell_labels = []
        cell_dataset = []
        cell_file_names = []
        cell_mask_id = []
        cell_statistics = []

        mask_ids = np.unique(mask)

        if import_limit == 'None' or import_limit > len(mask_ids):
            import_limit = len(mask_ids)
        else:
            import_limit = int(import_limit)

        for i in range(import_limit):

            if mask_ids[i] != 0:

                cell_mask = np.zeros(mask.shape, dtype=np.uint8)

                cell_mask[mask==mask_ids[i]] = 255

                contours, hierarchy = cv2.findContours(cell_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                cnt = contours[0]

                x,y,w,h = cv2.boundingRect(cnt)
                y1,y2,x1,x2 = y,(y+h),x,(x+w)

                cell_mask_crop = cell_mask[y1:y2,x1:x2]

                cell_image_crop = rgb[:,y1:y2,x1:x2].copy()

                stats = get_stats(image_channels, rgb, mask, cell_mask, cell_image_crop, cell_mask_crop, cnt)

                if mask_background:
                    cell_image_crop[:,cell_mask_crop==0] = 0

                cell_image_crop = resize_image(image_size, h, w, cell_image_crop, colicoords, resize)

                cell_image_crop = normalize99(cell_image_crop)

                if (np.max(cell_image_crop) - np.min(cell_image_crop)) > 0:

                    cell_image_crop = rescale01(cell_image_crop)
                    cell_image_crop = cell_image_crop.astype(np.float32)

                    cell_images.append(cell_image_crop)

                    label = antibiotic_list.index(dat_antibiotic)

                    cell_labels.append(label)

                    cell_file_names.append(file_name)
                    cell_dataset.append(dat_dataset)
                    cell_mask_id.append(mask_ids[i])
                    cell_statistics.append(stats)
    except:
        cell_dataset, cell_images, cell_labels, cell_file_namesm, mask_ids, mask_ids = [],[],[],[],[], []
        print(traceback.format_exc())

    return cell_dataset, cell_images, cell_labels, cell_file_names, cell_mask_id, cell_statistics



def cache_data(data, image_size, antibiotic_list, channel_list, cell_list,
               import_limit = 100, colicoords = False, mask_background = False, normalise = True, resize=False):

    data = data.sort_values(by=['segmentation_file']).reset_index(drop=True)
        
    data = data.groupby(["segmentation_file","content","folder"])
    
    data = [data.get_group(list(data.groups)[i]) for i in range(len(data))]
    
    print(f"loading {len(data)} images from AKGROUP into memory with image channels: {channel_list}")

    with Pool(2) as pool:
        
        results = pool.map(partial(get_cell_images,
                                    image_size=image_size,
                                    channel_list = channel_list,
                                    cell_list = cell_list,
                                    antibiotic_list = antibiotic_list,
                                    import_limit = import_limit,
                                    colicoords = colicoords,
                                    mask_background = mask_background,
                                    resize=resize), data)
        
        dataset, images, labels, file_names, mask_ids, stats = zip(*results)

        dataset = [item for sublist in dataset for item in sublist]
        images = [item for sublist in images for item in sublist]
        labels = [item for sublist in labels for item in sublist]
        file_names = [item for sublist in file_names for item in sublist]
        mask_ids = [item for sublist in mask_ids for item in sublist]
        stats = [item for sublist in stats for item in sublist]

    cached_data = dict(dataset=dataset,
                        images=images,
                        labels=labels,
                        file_names=file_names,
                        antibiotic_list = antibiotic_list,
                        mask_ids = mask_ids,
                        stats = stats)
     
    return cached_data



def shuffle_train_data(train_data):
      
    dict_names = list(train_data.keys())     
    dict_values = list(zip(*[value for key,value in train_data.items()]))
    
    random.shuffle(dict_values)
    
    dict_values = list(zip(*dict_values))
    
    train_data = {key:list(dict_values[index]) for index,key in enumerate(train_data.keys())}
    
    return train_data
                    

def limit_train_data(train_data, num_files):
    
    for key,value in train_data.items():
        
        train_data[key] = value[:num_files]
        
    return train_data








def get_training_data(cached_data, shuffle=True, ratio_train = 0.8, val_test_split=0.5, label_limit = "None", balance = False):

    label_names = cached_data.pop("antibiotic_list")

    train_data = {}
    val_data = {}
    test_data = {}
    
    cached_data = shuffle_train_data(cached_data)
    
    dataset = cached_data["dataset"]
    train_indices = np.argwhere(np.array(dataset)=="train")[:,0].tolist()
    test_indices = np.argwhere(np.array(dataset)=="test")[:,0].tolist()

    overlap = list(set(train_indices) & set(test_indices))

    data_sort = pd.DataFrame(cached_data).drop(labels = ["images"], axis=1)
    data_sort = data_sort.groupby(["labels"])
    
    for i in range(len(data_sort)):
    
        data = data_sort.get_group(list(data_sort.groups)[i])

        if shuffle is True:
            data = data.sample(frac=1, random_state=42)
            
        train_indcies = data[data["dataset"] == "train"].index.values.tolist()
        test_indcies = data[data["dataset"] == "test"].index.values.tolist()
        
        
        train_indices, val_indices = train_test_split(train_indices,
                                                      train_size=ratio_train,
                                                      random_state=42,
                                                      shuffle=True)
        
        if len(test_indcies) == 0:
            
            val_indices, test_indices = train_test_split(val_indices,
                                                         train_size=val_test_split,
                                                         random_state=42,
                                                         shuffle=True)

        overlap = list(set(train_indices) & set(val_indices) & set(test_indices))

        for key,value in cached_data.items():
    
            if key in ["images","masks"]:
               
                train_dat = list(np.array(value)[train_indices])
                val_dat = list(np.array(value)[val_indices])
                test_dat = list(np.array(value)[test_indices])
                
            else:
               
                train_dat = np.take(np.array(value), train_indices).tolist()
                val_dat = np.take(np.array(value), val_indices).tolist()
                test_dat = np.take(np.array(value), test_indices).tolist()
             
            if label_limit != "None":
                
                train_dat = train_dat[:label_limit]
                val_dat = val_dat[:label_limit]
                test_dat = test_dat[:label_limit]    
                  
            if key in train_data.keys():
          
              train_data[key].extend(train_dat)
              val_data[key].extend(val_dat)
              test_data[key].extend(test_dat)
      
            else:
              
              train_data[key] = train_dat
              val_data[key] = val_dat
              test_data[key] = test_dat
        
    if shuffle == True:
        print("shuffling train/val/test datasets")
        train_data = shuffle_train_data(train_data)    
        val_data = shuffle_train_data(val_data)
        test_data = shuffle_train_data(test_data)

    if balance == True:
        print("balancing train/val/test datasets")
        train_data = balance_dataset(train_data)
        val_data = balance_dataset(val_data)
        test_data = balance_dataset(test_data)

    train_data["antibiotic_list"] = label_names
    val_data["antibiotic_list"] = label_names
    test_data["antibiotic_list"] = label_names

    return train_data, val_data, test_data  


def balance_dataset(dataset):

    data_sort = dataset["labels"]
    unique, counts = np.unique(data_sort, return_counts=True)

    max_count = np.min(counts)

    balanced_dataset = {}

    for unique_key in unique:

        unique_indices = np.argwhere(np.array(data_sort) == unique_key)[:,0].tolist()[:max_count]

        for key, value in dataset.items():

            if key not in balanced_dataset.keys():
                    balanced_dataset[key] = []

            balanced_dataset[key].extend([value[index] for index in unique_indices])

    return balanced_dataset



