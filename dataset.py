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


def get_dataloaders(DATA_PATH=DATA_PATH, IMAGE_SIZE=IMAGE_SIZE, BATCH_SIZE=BATCH_SIZE):
    TRAIN = 'train'
    TEST = 'test'
    VAL = 'val'
    DATASET_TYPE = [TRAIN, VAL, TEST]
    # train_data = datasets.ImageFolder(os.path.join(DATA_PATH, TRAIN), 
    #                                 transform = transforms.ToTensor())
    # already_calculated = True
    # if already_calculated == True:
    #     means = torch.tensor([0.1934, 0.1934, 0.1934])
    #     stds = torch.tensor([0.2013, 0.2013, 0.2013])
    # else:
    #     means = torch.zeros(3)
    #     stds = torch.zeros(3)

    #     for img, _ in train_data:
    #         means += torch.mean(img, dim = (1,2))
    #         stds += torch.std(img, dim = (1,2))

    #     means /= len(train_data)
    #     stds /= len(train_data)

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


# print("ABCA")
# if __name__=="__main__":
#     print("!@#!@#")
#     get_dataloaders()