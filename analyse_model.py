# From model, load classifications and correlate with properties

import torch
import re
import pandas as pd

MODEL_PATH = r"C:\Users\farrara\Desktop\AMR_Pytorch_Classification\models\AntibioticClassification_SensResB_240916_2017\AMRClassification_[Resistant-Cy3]_240916_2017"

def extract_number_after_L(file_name):
    # Define the regular expression pattern to find the number after '_L'
    match = re.search(r'_L(\d+)', file_name)
    # If a match is found, return the number
    if match:
        return int(match.group(1))
    else:
        # Return None or handle the case where '_L' is not found
        return None

# device
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

model = torch.load(MODEL_PATH, map_location=device, weights_only=False)

test_stats = model.get("test_stats", [])

# Extracting the "File Name" values
file_names = [item.get("File Name", "Unknown") for item in test_stats]

strains = [extract_number_after_L(name) for name in file_names]

model_analysis = pd.DataFrame(
    {'strain': strains,
     'pred label': model['pred_labels']
     })

model_analysis.to_csv('SensResModel_TestB.csv')

