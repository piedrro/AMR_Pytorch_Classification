
from torch.utils import data
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from trainer import Trainer
from file_io import get_metadata, get_cell_images, cache_data, get_training_data
from dataloader import load_dataset
import pickle

image_size = (64,64)
resize = False

antibiotic_list = ["Untreated", "Ciprofloxacin"]
microscope_list = ["BIO-NIM", "ScanR"]
channel_list = ["Cy3"]
cell_list = ["single"]
train_metadata = {"content": "[E.Coli MG1655]",
                  "antibiotic concentration": ["0XEUCAST","1XEUCAST"],
                  "segmentation_curated":True}
test_metadata = {"user_meta3": "BioRepA",
                 "antibiotic concentration": ["0XEUCAST", "1XEUCAST"],
                 "user_meta1": ["L17667"]}

model_backbone = 'efficientnet_b0'
ratio_train = 0.9
val_test_split = 0.5
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 100
AUGMENT = True

#AKSEG_DIRECTORY = r"/run/user/26441/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Piers/AKSEG/"
AKSEG_DIRECTORY = r"\\physics\dfs\DAQ\CondensedMatterGroups\AKGroup\Piers\AKSEG\"
USER_INITIAL = "AF"

SAVE_DIR = "/home/farrara/Code/AMR_Pytorch/"
MODEL_FOLDER_NAME = "AntibioticClassificationTest"



# device
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


akseg_metadata = get_metadata(AKSEG_DIRECTORY,
                              USER_INITIAL,
                              channel_list,
                              antibiotic_list,
                              microscope_list,
                              train_metadata,
                              test_metadata)

#
if __name__ == '__main__':

    cached_data = cache_data(akseg_metadata,
                              image_size,
                              antibiotic_list,
                              channel_list,
                              cell_list,
                              import_limit = 9999,
                              mask_background=True,
                              resize=resize)

    with open('cacheddata.pickle', 'wb') as handle:
        pickle.dump(cached_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('cacheddata.pickle', 'rb') as handle:
        cached_data = pickle.load(handle)

    num_classes = len(np.unique(cached_data["labels"]))

    train_data, val_data, test_data = get_training_data(cached_data,
                                                          shuffle=True,
                                                          ratio_train = 0.9,
                                                          val_test_split=0.5,
                                                          label_limit = 'None')

    training_dataset = load_dataset(images = train_data["images"],
                                    labels = train_data["labels"],
                                    num_classes = num_classes,
                                    augment=AUGMENT)

    validation_dataset = load_dataset(images = val_data["images"],
                                      labels = val_data["labels"],
                                      num_classes = num_classes,
                                      augment=False)

    test_dataset = load_dataset(images = test_data["images"],
                                labels = test_data["labels"],
                                num_classes = num_classes,
                                augment=False)
    trainloader = data.DataLoader(dataset=training_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    valoader = data.DataLoader(dataset=validation_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    # Preview images
    # images, labels = next(iter(trainloader))
    #
    # for img in images:
    #
    #     plt.imshow(img[0])
    #     plt.show()

    model_path = r"\\physics\dfs\DAQ\CondensedMatterGroups\AKGroup\Alison\AntibioticClassification_TestL17667_CipRedo_230426_2202\AMRClassification_[Ciprofloxacin-Cy3]_230426_2202"

    model = models.densenet121(num_classes=len(antibiotic_list)).to(device)
    model = model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")

    trainer = Trainer(model=model,
                      device=device,
                      optimizer=optimizer,
                      criterion=criterion,
                      trainloader=trainloader,
                      valoader=valoader,
                      lr_scheduler=scheduler,
                      tensorboard=True,
                      antibiotic_list = antibiotic_list,
                      channel_list = channel_list,
                      epochs=EPOCHS,
                      batch_size = BATCH_SIZE,
                      model_folder_name = MODEL_FOLDER_NAME)

    model_data = trainer.evaluate(model,
                                  model_path,
                                  train_data["images"],
                                  test_data["images"],
                                  test_data["labels"],
                                  len(antibiotic_list))
    torch.cuda.empty_cache()
