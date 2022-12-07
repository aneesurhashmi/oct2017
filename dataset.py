from __future__ import print_function, division
import os
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from collections import Counter


# Data Parameters
DATA_PATH = "../original_data/"
BATCH_SIZE = 2000
IMAGE_SIZE = 150
# IMAGE_SIZE = 224 # changing for Imagenet ViT
NUM_WORKERS = 26


def get_dataloaders(DATA_PATH=DATA_PATH, IMAGE_SIZE=IMAGE_SIZE, BATCH_SIZE=BATCH_SIZE, load_augmented_dataset= False):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    DATASET_TYPE = [TRAIN, VAL, TEST]

    data_transforms = {
        TRAIN: transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]),
        VAL: transforms.Compose([
            transforms.Resize(((IMAGE_SIZE, IMAGE_SIZE))),
            transforms.ToTensor(),
        ]),
        TEST: transforms.Compose([
            transforms.Resize(((IMAGE_SIZE, IMAGE_SIZE))),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(
            os.path.join(DATA_PATH, x), 
            transform=data_transforms[x]
        )
        for x in DATASET_TYPE
    }
    # set true when using augmented dataset
    if load_augmented_dataset:
        dataset_sizes = {x: len(image_datasets[x]) for x in DATASET_TYPE}

        # print dataset size, class
        for x in DATASET_TYPE:
            print("Loaded {}  {} images".format(dataset_sizes[x], x))

        # dataloaders
        train_loader = torch.utils.data.DataLoader(
            image_datasets[TRAIN], 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            # sampler=train_weighted_sampler,
        #     shuffle=True, 
            num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            image_datasets[TEST], 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4
        )

        val_loader = torch.utils.data.DataLoader(
            image_datasets[VAL], 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4
        )
        return {"train_loader":train_loader, "val_loader":val_loader, "test_loader":test_loader}
    # for weighted random sampling
    else:
        dataset_sizes = {x: len(image_datasets[x]) for x in DATASET_TYPE}
        # computing class weights for imbalanced data set
        train_targets = [sample[1] for sample in image_datasets[TRAIN].imgs]
        counter = Counter(train_targets)
        # print(set(train_targets))
        class_count = [i for i in counter.values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        train_samples_weight = [class_weights[class_id] for class_id in train_targets]
        train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_samples_weight, dataset_sizes[TRAIN])
        # print dataset size, class
        for x in DATASET_TYPE:
            print("Loaded {}  {} images".format(dataset_sizes[x], x))

        # dataloaders
        train_loader = torch.utils.data.DataLoader(
            image_datasets[TRAIN], 
            batch_size=BATCH_SIZE, 
            sampler=train_weighted_sampler,
        #     shuffle=True, 
            num_workers=NUM_WORKERS
        )

        test_loader = torch.utils.data.DataLoader(
            image_datasets[TEST], 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS

        )

        val_loader = torch.utils.data.DataLoader(
            image_datasets[VAL], 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=NUM_WORKERS
        )
        return {"train_loader":train_loader, "val_loader":val_loader, "test_loader":test_loader}




from time import time
import multiprocessing as mp

# function to check optimal number of workers for dataloader
def optim_num_worker ():
    for num_workers in range(2, mp.cpu_count(), 2):  
        TRAIN = 'train'
        TEST = 'test'
        VAL = 'val'
        DATASET_TYPE = [TRAIN]

        data_transforms = {
            TRAIN: transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
            ]),
        }
        image_datasets = {
            x: datasets.ImageFolder(
                os.path.join(DATA_PATH, x), 
                transform=data_transforms[x]
            )
            for x in DATASET_TYPE
        }

        dataset_sizes = {x: len(image_datasets[x]) for x in DATASET_TYPE}
        # computing class weights for imbalanced data set
        train_targets = [sample[1] for sample in image_datasets[TRAIN].imgs]
        counter = Counter(train_targets)
        # print(set(train_targets))
        class_count = [i for i in counter.values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float)
        train_samples_weight = [class_weights[class_id] for class_id in train_targets]
        train_weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            train_samples_weight, dataset_sizes[TRAIN])
        # print dataset size, class
        for x in DATASET_TYPE:
            print("Loaded {}  {} images".format(dataset_sizes[x], x))

        # dataloaders
        train_loader = torch.utils.data.DataLoader(
            image_datasets[TRAIN], 
            batch_size=64, 
            sampler=train_weighted_sampler,
        #     shuffle=True, 
            num_workers=num_workers
        )
        # train_loader = DataLoader(train_reader,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
        start = time()
        for epoch in range(1, 3):
            for i, _ in enumerate(train_loader, 0):
                pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))



if __name__=="__main__":
    print("Checking number of workers...")
    optim_num_worker()