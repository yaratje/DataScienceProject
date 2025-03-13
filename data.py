""""
All stuff related to data
"""

import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

""""
Setup basic variables, get dataset from local folder, create dataloader, and extract labels and images for later use
"""
def setup():
    #TRANSFORM from assignment 2
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    global dataset
    dataset = ImageFolder(root="./Incidents-subset", transform=transform)
    global dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    global labels 
    labels = np.array(dataset.targets)
    global images
    images = np.array([img for img, label in dataset])



""""
Function that splits dataset into train and test datasets
@dataset, the entire dataset
@ratio, ratio beteen 0-1 what the split should be
"""
def split_data(dataset, ratio):
    #stratify=labels = balanced labels distribution :O
    train_data_x,test_data_x,train_data_y,test_data_y = train_test_split(images,labels, test_size=ratio, stratify=labels)

    return train_data_x,test_data_x,train_data_y,test_data_y
    
train_data_x, test_data_x , train_data_y, test_data_y = split_data(dataset, 0.2)