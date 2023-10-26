import pandas as pd
import numpy as np
import sys
import os
import sys
import torch
import torch.nn as nn

import timm

class timm_model(nn.Module):

    def __init__(self, num_classes, backbone='densenet121', pretrained=False, pretained_path=""):
        
        super(timm_model, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.pretrained_path = pretained_path

        timm_pretrained_models = timm.list_models(pretrained=True)
        
        if self.pretrained==True:
            if backbone not in timm_pretrained_models:
                print(f"Pretained model not available for {backbone}, loading untrained model")
                self.pretrained = False

        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=self.num_classes) 
        
        if self.pretrained_path != "":
            model_data = torch.load(self.pretrained_path)
            self.backbone = self.backbone.load_state_dict(model_data['model_state_dict'])
        
    def forward(self, x, labels=None):
        
        out = self.backbone(x)
        
        return out
    
timm_pretrained_models = timm.list_models(pretrained=False)   

# model = timm_model(2, pretrained=True)

# model.eval()

# state_dict = model.state_dict()

# model = timm_model(2, pretrained=False)

# model.load_state_dict(state_dict)


    
    
    
    
    
    
    
    