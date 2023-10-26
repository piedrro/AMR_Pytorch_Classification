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

        from albumentations.augmentations.blur.transforms import Blur
        from albumentations.augmentations.transforms import RGBShift, GaussNoise, PixelDropout, ChannelShuffle
        from albumentations import RandomBrightnessContrast, RandomRotate90, Flip, Affine
        
        """applies albumentations augmentation to image and mask, including resizing the images/mask to the crop_size"""


        shift_channels = A.Compose([Affine(translate_px=[-5,5])])

        mask = img[0].copy()

        for i, chan in enumerate(img):
            if i != 0:
                chan = shift_channels(image=chan)['image']
                chan[mask==0] = 0
                img[i] = chan


        img = np.moveaxis(img,0,-1)

        # geometric transforms
        transform = A.Compose([
            RandomRotate90(),
            Flip(),
            Affine(scale=(0.6,1.4),shear=(-20,20),rotate=(-360,360),translate_px=[-20,20]),
        ])

        img = transform(image=img)['image']
        mask = img.copy()

        #pixel transforms
<<<<<<< HEAD
        # AF trying var_limit 50 instead of 0.01
        #A.augmentations.transforms.GaussNoise(var_limit=0.01, per_channel=True),
        transform = A.Compose([A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, always_apply=False, p=0.5),
                               A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5)])
        
=======
        transform = A.Compose([
            GaussNoise(var_limit=0.0005, per_channel=True, always_apply=False),
            Blur(blur_limit=5, always_apply=False, p=0.5),
            RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.5, always_apply=False),
            PixelDropout(dropout_prob=0.05, per_channel=True, p=0.5),
        ])

>>>>>>> upstream/main
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
    
    
    
    
    
    