import pandas as pd
import torch
import timm
import numpy as np
<<<<<<< HEAD
from matplotlib import pyplot as plt
from datetime import datetime
=======

>>>>>>> upstream/main
from trainer import Trainer
from file_io import get_metadata, get_training_data, cache_data
import pickle
<<<<<<< HEAD
from evaluate import evaluate_model

image_size = (64,64)
resize = False

<<<<<<< Updated upstream
antibiotic_list = ["Untreated", "Gentamicin"]
=======
antibiotic_list = ["Ciprofloxacin"]
>>>>>>> Stashed changes
microscope_list = ["BIO-NIM", "ScanR"]
channel_list = ["Cy3"]
cell_list = ["single"]
train_metadata = {"content": "E.Coli MG1655", "segmentation_curated":True}
<<<<<<< Updated upstream
test_metadata = {"user_meta3": "BioRepC"}
=======
test_metadata = {"user_meta1": "L17667", "segmentation_curated": True}
>>>>>>> Stashed changes
=======


image_size = (64,64)
resize = False
antibiotic_list = ["Untreated", "Ciprofloxacin"]
microscope_list = ["BIO-NIM"]
channel_list = ["532","405"]
cell_list = ["single"]
train_metadata = {"user_meta1":"2021 DL Paper", "user_meta2":"Lab Strains"}
test_metadata = {"user_meta1":"2021 DL Paper", "user_meta2":"Lab Strains", "user_meta3":"Repeat 7"}
>>>>>>> upstream/main

model_backbone = 'densenet121'
model_backbone = 'efficientnet_b0'

ratio_train = 0.9
val_test_split = 0.5
<<<<<<< HEAD
BATCH_SIZE = 100
LEARNING_RATE = 0.001
EPOCHS = 100
AUGMENT = True

AKSEG_DIRECTORY = r"/run/user/26441/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Piers/AKSEG/"
=======
BATCH_SIZE = 10
LEARNING_RATE = 0.01
EPOCHS = 10
AUGMENT = True

AKSEG_DIRECTORY = r"/run/user/26623/gvfs/smb-share:server=physics.ox.ac.uk,share=dfs/DAQ/CondensedMatterGroups/AKGroup/Piers/AKSEG"
# AKSEG_DIRECTORY = r"\\physics\dfs\DAQ\CondensedMatterGroups\AKGroup\Piers\AKSEG"
>>>>>>> upstream/main


USER_INITIAL = "AZ"

SAVE_DIR = "/home/turnerp/PycharmProjects/AMR_Pytorch_Classification"
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

<<<<<<< HEAD
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


    model = models.densenet121(num_classes=len(antibiotic_list)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
=======
    print(f"train_data: {len(train_data['images'])}, val_data: {len(val_data['images'])}, test_data: {len(test_data['images'])}")
    #
    model = timm.create_model(model_backbone, pretrained=True, num_classes=len(antibiotic_list)).to(device)
    # 'timm.list_models()' to list available models
>>>>>>> upstream/main

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

<<<<<<< HEAD
    model_data = trainer.evaluate(model,
                                  model_path,
                                  train_data["images"],
                                  test_data["images"],
                                  test_data["labels"],
                                  len(antibiotic_list))
    torch.cuda.empty_cache()
=======
    trainer.evaluate(model_path)

    # model_path = r"models/AntibioticClassification_230324_1832/[Ciprofloxacin-532-405]/AMRClassification_[Ciprofloxacin-532-405]_230324_1832"

>>>>>>> upstream/main
