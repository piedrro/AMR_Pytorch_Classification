# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 14:52:47 2024

@author: farrara
"""
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from natsort import natsorted
import scipy.stats as stats

MODEL_PATH = r'models/AntibioticClassification_TimeLapse_CipModel_1XCip5min_240805_1636/AMRClassification_[Ciprofloxacin-Cy3]_231118_1123'

def get_stats(PATH):

    model=torch.load(MODEL_PATH)

    model_stats = []

    for i in range(len(model['pred_labels'])):
        cell = {}
        cell['pred_label'] = model['pred_labels'][i]
        file_name = model['test_stats'][i]['File Name']
        time = re.search(r'_t(\d+)_', file_name)
        cell['time'] = (int(time.group(1))*5)+5
        cell['area'] = model['test_stats'][i]['Cell Area (Pixels)']
        cell['mask'] = model['test_stats'][i]['Mask ID'] # can be used with cv2 to plot overlay
        model_stats.append(cell)
        print(i)
    
    model_stats = pd.DataFrame(model_stats)
    
    return model_stats

df = get_stats(MODEL_PATH)
df.to_csv('out_cip1xEUCAST_A_cip.csv', index=False)
'''rest can be done in origin from this csv'''
# Stats on columns -> mean and standard error by time