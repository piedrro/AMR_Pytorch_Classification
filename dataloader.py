from torch.utils import data
import torch
import numpy as np
from skimage import exposure
import albumentations as A
import torch.nn.functional as F
from torchvision.transforms import Normalize


class load_dataset(data.Dataset):

    def __init__(self,
                 images = [],
                 labels = [],
                 treatment_labels = [],
                 num_classes = int,
                 augment=None,
                 ):


        self.augment = augment
        self.images = images
        self.labels = labels
        self.num_classes = num_classes
                    

    def __len__(self):
        return len(self.images)

    def augment_images(self, img):
        
        """applies albumentations augmentation to image and mask, including resizing the images/mask to the crop_size"""
        
        img = np.moveaxis(img,0,-1)



        # geometric transforms
        transform = A.Compose([A.RandomRotate90(),
                                A.Flip(),
                                A.augmentations.geometric.transforms.Affine(scale=(0.7,1.3),
                                                                            shear=(-20,20),
                                                                            rotate=(-90,90),
                                                                            translate_px=[-20,20])])
        
        img = transform(image=img)['image']
        mask = img.copy()
        
        #pixel transforms
        # AF trying var_limit 50 instead of 0.01
        transform = A.Compose([A.augmentations.transforms.GaussNoise(var_limit=0.01, per_channel=True),
                               A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5)])
        
        img = transform(image=img)['image']
        img[mask==0] = 0
                
        img = np.moveaxis(img,-1,0)
        
        return img


    def normalize99(self, X):
        """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
        X = X.copy()
        v_min, v_max = np.percentile(X[X!=0], (1, 99))
        X = exposure.rescale_intensity(X, in_range=(v_min, v_max))
        return X

    def rescale01(self, x):
        """ normalize image from 0 to 1 """
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x


    def postprocess(self, x, y):
        """re-formats the image/masks for training"""

        # Typecasting
        x = torch.from_numpy(x.copy()).float()
        y = F.one_hot(torch.tensor(y), num_classes=self.num_classes).float()

        return x, y

    def __getitem__(self, index: int):

        image, label = self.images[index], self.labels[index]
        
        image = image.astype(np.float32)
                
        if self.augment:
            image = self.augment_images(image)
        
        image, label = self.postprocess(image, label)
        
        return image, label
    
    
    
    
    
    