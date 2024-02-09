"""
Adapted to test only, manual 2 classes
"""

import torch
import timm
import numpy as np
from trainer import Trainer
from file_io import get_metadata, get_cell_images, cache_data, get_training_data
import pickle

image_size = (64,64)
resize = False

#antibiotic_list = ["Untreated", "Ciprofloxacin"]
antibiotic_list = ["Ciprofloxacin"]
microscope_list = ["BIO-NIM"]
channel_list = ["Cy3"]
cell_list = ["single"]
train_metadata = {"content": "E.Coli Clinical"}
test_metadata = {"content": "E.Coli Clinical",
                 "antibiotic concentration": "1XEUCAST",
                 "user_meta1": "L48480",
                 "user_meta3": "BioRepA"} # this tag includes only 0XEUCAST, 1XEUCAST, 20X, and none abx conc

model_backbone = 'efficientnet_b0'

ratio_train = 0.9
val_test_split = 0.5
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 10
AUGMENT = True

## Directory Linux
#AKSEG_DIRECTORY = r"/run/user/26623/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Piers/AKSEG"
## Directory Windows
AKSEG_DIRECTORY = r"\\physics\dfs\DAQ\CondensedMatterGroups\AKGroup\Piers\AKSEG"

USER_INITIAL = "AF"

#SAVE_DIR = "/home/turnerp/PycharmProjects/AMR_Pytorch_Classification"
SAVE_DIR = r"H:\code\AMR_PyTorch"
MODEL_FOLDER_NAME = "AntibioticClassification"

MODEL_PATH = r"C:\Users\farrara\Desktop\AMR_Pytorch_Classification\AMRClassification_[Ciprofloxacin-Cy3]_231118_1123"

# device
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

akseg_metadata = get_metadata(AKSEG_DIRECTORY,
                              USER_INITIAL,
                              channel_list,
                              antibiotic_list,
                              microscope_list,
                              train_metadata,
                              test_metadata,)
akseg_metadata.to_csv('akseg_metadata.csv')

if __name__ == '__main__':

    cached_data = cache_data(
        akseg_metadata,
        image_size,
        antibiotic_list,
        channel_list,
        cell_list,
        import_limit = 'None',
        mask_background=True,
        resize=resize)

    with open('cacheddata.pickle', 'wb') as handle:
        pickle.dump(cached_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cacheddata.pickle', 'rb') as handle:
        cached_data = pickle.load(handle)

    #num_classes = len(np.unique(cached_data["labels"]))
    num_classes = 2 # Treated and untreated
    print(f"num_classes: {num_classes}, num_images: {len(cached_data['images'])}")
    # Balance False for testing
    train_data, val_data, test_data = get_training_data(cached_data,
                                                          shuffle=True,
                                                          ratio_train = 0.8,
                                                          val_test_split=0.5,
                                                          label_limit = 'None',
                                                          balance = False,)

    print(f"train_data: {len(train_data['images'])}, val_data: {len(val_data['images'])}, test_data: {len(test_data['images'])}")
    #
    model = timm.create_model(model_backbone, pretrained=True, num_classes=2).to(device)
    # 'timm.list_models()' to list available models
    model_state_dict = torch.load(MODEL_PATH)['model_state_dict']
    model.load_state_dict(model_state_dict, strict=False)
    #model = torch.load(MODEL_PATH)
    print(type(model))
    antibiotic_list = ["Untreated", "Ciprofloxacin"]
    trainer = Trainer(model=model,
                      num_classes=num_classes,
                      augmentation=AUGMENT,
                      device=device,
                      learning_rate=LEARNING_RATE,
                      train_data=train_data,
                      val_data=val_data,
                      test_data=test_data,
                      tensorboard=True,
                      antibiotic_list = antibiotic_list,
                      channel_list = channel_list,
                      epochs=EPOCHS,
                      batch_size = BATCH_SIZE,
                      model_folder_name = MODEL_FOLDER_NAME)

    trainer.plot_descriptive_dataset_stats(show_plots=False, save_plots=True)
    #
    model_data = trainer.evaluate(MODEL_PATH)
    torch.cuda.empty_cache()