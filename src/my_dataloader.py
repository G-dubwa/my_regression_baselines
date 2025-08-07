import os
import torch
import numpy as np
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt

TTP_FILE = "data/cage/TimeToPositivityDataset.csv"

def get_data(dataset, data_folds, dev_fold, test_fold, image_folder, loss, batch_size, num_outer_folds=10):
    train_data = None
    val_data = None
    test_data = None
    
    training_set_fold_files = []
    for fold in range(num_outer_folds):
        if fold != dev_fold and fold != test_fold:
            training_set_fold_files.append(data_folds+"/fold_"+str(fold))
    
    dev_set_fold_file = data_folds+"/fold_"+str(dev_fold)
    test_set_fold_file = data_folds+"/fold_"+str(test_fold)

    mean, std = get_mean_std(training_set_fold_files, dataset, image_folder)
    
    train_dataset = ConcatDataset([CoughDatasetCleaned(dataset, file+".csv",image_folder, loss, mean=mean, std=std) for file in training_set_fold_files])
    train_data = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)

    # Get all the development/validation data using a dataloader class if development data is not None
    if not dev_fold is None:
        val_data_set = CoughDatasetCleaned(dataset, dev_set_fold_file+".csv", image_folder, loss, mean=mean, std=std)
        val_data = DataLoader(val_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: val_data = None

    # Get all the test data using a dataloader class if development data is not None
    if not test_fold is None:
        test_data_set = CoughDatasetCleaned(dataset, test_set_fold_file+".csv", image_folder, loss, mean=mean, std=std)
        test_data = DataLoader(test_data_set, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
    else: test_data = None

    return train_data, val_data, test_data


def get_mean_std(training_set_fold_files, dataset, image_folder):
    training_dataset = EmptyDataset()
    for file in training_set_fold_files:
        training_dataset = ConcatDataset([training_dataset, CoughDataset(dataset, file+".csv", image_folder)])

    mels = []
    train_loader = DataLoader(training_dataset, batch_size=100, num_workers=1)
    for images, _ in train_loader:
        mels.append(images)
    
    mels = torch.cat(mels, dim=0)
    perc = int(2/3*mels.shape[1])
    mean = torch.mean(mels[:,:perc,:],dim=(0,1))
    std = torch.std(mels[:,:perc,:],dim=(0,1))

    return mean, std


class EmptyDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, index):
        raise IndexError("Empty dataset cannot be indexed")

class CoughDataset(Dataset):
    def __init__(self, dataset, annotated_fold_file, image_folder):
        self.targets = pd.read_csv(annotated_fold_file)
        self.image_folder = image_folder
        self.dataset = dataset
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self,idx):
        target = self.targets["Status"][idx]
        path_to_image = os.path.join(self.image_folder, str(self.targets["Cough_ID"][idx])+".npy")
        image = torch.tensor(np.transpose(np.load(path_to_image)))
        image = image[:50,:]

        if self.dataset == "cage" and image.shape[0]<50: image = torch.nn.functional.pad(image, (0,0,(50-image.shape[0]),0),"constant",0)

        return image, target

class CoughDatasetCleaned(Dataset):
    def __init__(self, dataset, annotated_fold_file, image_folder, loss, mean, std, filter_invalid=True):
        
        self.coughs = pd.read_csv(annotated_fold_file)

        
        required_cols = {"Cough_ID", "Status", "Time_to_positivity"}
        missing_cols = required_cols - set(self.coughs.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in {annotated_fold_file}: {missing_cols}")

        
        if filter_invalid:
            self.coughs = self.coughs[~((self.coughs["Time_to_positivity"] == -1))] # (self.coughs["Status"] == 1) & 

        
        self.coughs = self.coughs.dropna(subset=["Time_to_positivity"]).reset_index(drop=True)

        
        self.image_folder = image_folder
        self.loss = loss
        self.mean = mean
        self.std = std
        self.dataset = dataset
    
    def __len__(self):
        return len(self.coughs)
    
    def __getitem__(self, idx):
        if self.dataset == "cage":
            label = self.coughs["Time_to_positivity"][idx]
            path_to_image = os.path.join(self.image_folder,str(self.coughs["Cough_ID"][idx])+".npy")
            image_raw = torch.tensor(np.transpose(np.load(path_to_image)))

            if self.dataset == "cage" and image_raw.shape[0]<40: image_raw = torch.nn.functional.pad(image_raw, (0,0,(40-image_raw.shape[0]),0),"constant",image_raw.min())

            if self.loss == "mse":
                image = self.standardize(image_raw)
                image_mean = image.mean(0)
                return image_mean, label

            elif self.loss == "mse_cnn":
                return self.pad(self.standardize(self.repeat(image_raw))), label

    def repeat(self, image):
        image = image[None,:,:]
        image = torch.concat((image, image, image))
        return image

    def pad(self, image):
        if (image.shape[-1] < 224) or (image.shape[-2] < 224):
            pad_width = ((0,0),(0,224-image.shape[-2]),(0,224-image.shape[-1]))
            image = np.pad(image, pad_width=pad_width, constant_values=0)
        return image

    def standardize(self, image):
        image = (image - self.mean)/self.std
        return image
    
    

    

