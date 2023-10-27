"""
STILL NEEDS TO BE ADAPTED TO TEST ONLY
"""

import torch
import timm
import numpy as np
from trainer import Trainer
from file_io import get_metadata, get_cell_images, cache_data, get_training_data
import pickle

image_size = (64,64)
resize = False

antibiotic_list = ["Ciprofloxacin"]
microscope_list = ["BIO_NIM"]
channel_list = ["Cy3"]
cell_list = ["single"]
train_metadata = {"content": "E.Coli MG1655",
                  "segmentation_curated": True}
test_metadata = {"user_meta1": "L17667",
                 "segmentation_curated": True,
                 "user_meta3": "BioRepA"}

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
SAVE_DIR = r"\\cmfs1.physics.ox.ac.uk\cm\farrara\code\AMR_PyTorch"
MODEL_FOLDER_NAME = "AntibioticClassification"

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

    num_classes = len(np.unique(cached_data["labels"]))

    print(f"num_classes: {num_classes}, num_images: {len(cached_data['images'])}")

    train_data, val_data, test_data = get_training_data(cached_data,
                                                          shuffle=True,
                                                          ratio_train = 0.8,
                                                          val_test_split=0.5,
                                                          label_limit = 'None',
                                                          balance = True,)

    print(f"train_data: {len(train_data['images'])}, val_data: {len(val_data['images'])}, test_data: {len(test_data['images'])}")
    #
    model = timm.create_model(model_backbone, pretrained=True, num_classes=len(antibiotic_list)).to(device)
    # 'timm.list_models()' to list available models

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

    trainer.visualise_augmentations(n_examples=10, show_plots=False, save_plots=True)

    trainer.tune_hyperparameters(num_trials=50, num_images = 2000, num_epochs = 10)

    model_path = trainer.train()

    trainer.evaluate(model_path)

    # model_path = r"models/AntibioticClassification_230324_1832/[Ciprofloxacin-532-405]/AMRClassification_[Ciprofloxacin-532-405]_230324_1832"