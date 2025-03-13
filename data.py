""""
All stuff related to data
"""

import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from collections import Counter
from PIL import Image



def tryloader(path):
    try:
        #see if the image is actually an image, also convert to RGB yaknow to be safe
        return Image.open(path).convert("RGB") 
    except Exception as e:
        print(f"This image is broken: {path} ({e})")
        return None

""""
Setup basic variables, get dataset from local folder, create dataloader, and extract labels and images for later use
"""
def setup():
    #TRANSFORM from assignment 2
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    global dataset
    dataset = ImageFolder(root="./Incidents-subset", transform=transform, loader=tryloader)
    valid_samples = []
    for path, label in dataset.samples:
        #check again?
        img = tryloader(path)
        if img is not None:
            valid_samples.append((path, label))
    #ensure only valid samples remain in the dataset
    dataset.samples = valid_samples
    dataset.targets = [label for _, label in valid_samples]
    
    global dataloader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    global labels 
    labels = np.array(dataset.targets)
    global images
    images = np.array([img for img, _ in dataset if img is not None])



""""
Function that splits dataset into train and test datasets
@dataset, the entire dataset
@ratio, ratio beteen 0-1 what the split should be (for this project set to 0.2/0.8 test/train)
"""
def split_data(dataset, ratio):
    #stratify=labels = balanced labels distribution :O
    train_data_x,test_data_x,train_data_y,test_data_y = train_test_split(images,labels, test_size=ratio, stratify=labels)

    return train_data_x,test_data_x,train_data_y,test_data_y
    

def class_counts():
    #count the occurances per label
    class_counts = Counter(labels)

    #print the outcome
    for class_idx, count in class_counts.items():
        print(f"{dataset.classes[class_idx]}: number of images {count}, class index: {class_idx}")


if __name__ == '__main__':
    setup()
    class_counts()
    train_data_x, test_data_x , train_data_y, test_data_y = split_data(dataset, 0.2)