""""
All stuff related to data
"""

import torch

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from collections import Counter
from PIL import Image
import json, os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import random


#Utility functions

def save_valid_samples(valid_samples, cache_file):
    with open(cache_file, "w") as f:
        json.dump(valid_samples, f)

def load_valid_samples(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            return json.load(f)
    return None

"""
Loader function used as loader in ImageFolder, ensures images path is vallid and converts image to rgb format and returns
If path is not valid (mostly files starting with .) then return None
@path, filepath where the dataset is located
"""

def tryloader(path):
    try:
        #see if the image is actually an image, also convert to RGB yaknow to be safe
        return Image.open(path).convert("RGB") 
    except Exception as e:
        print(f"This image is broken: {path} ({e})")
        return None


"""
Preprocess the images (load transform) and cache them as a tensor to save more time cause it takes to lonngg to transform every time.
"""  
def preprocess_tensor(dataset, cache_path):
    if os.path.exists(cache_path):
        print("Loading images from cache")
        return torch.load(cache_path)
    else:
        print("Preprocessing images and chacing")
        all_batches = []
        #process images in batches as my laptop does not like this stuff
        temp_loader = DataLoader(dataset, batch_size=251, shuffle=False, num_workers=2, pin_memory=True)
        for batch in temp_loader:
            imgs, _ = batch
            all_batches.append(imgs)
        #concatenate into one tensor
        images_tensor = torch.cat(all_batches, dim=0)
        torch.save(images_tensor, cache_path)
        return images_tensor


def balance_classes(dataset):
    original_counts = Counter(dataset.targets)
    highest = max(original_counts.values())

    #make copy
    #original samples + flag to know they are the original images
    balanced_list = []
    for path, label in dataset.samples:
        balanced_list.append((path, label, False))  # original image (no augmentation)

    #create a list of indices where the images are for each label.
    label_to_samples = {}
    for sample in balanced_list:
        label = sample[1]
        label_to_samples.setdefault(label, []).append(sample)

    #for each class, add images so classes are balanced, with true flag to know they need to be augmented.
    aug_counts = {}
    for label, count in original_counts.items():
        extra_needed = highest - count
        aug_counts[label] = extra_needed
        for _ in range(extra_needed):
            sample = random.choice(label_to_samples[label])
            balanced_list.append((sample[0], sample[1], True))
    

    return balanced_list, original_counts


def augi(batch):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    augment = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
    transforms.ToTensor()
    ])
    


    images = []
    labels = []
    for (path, label, flag) in batch:
        img = tryloader(path)
        if flag:
            img = augment(img)
        else:
            img = transform(img)
        images.append(img)
        labels.append(label)
    return torch.stack(images), torch.tensor(labels)


""""
Setup basic variables, get dataset from local folder, create dataloader, and extract labels and images for later use
Additionally, added caching of all the valid file paths as a json file cause filtering the dataset took long, so if file exists, load from there
"""

def setup():
    #TRANSFORM from assignment 2
    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageFolder(root="./Incidents-subset", transform=transform, loader=tryloader)

    cache_file = "valid_samples.json"
    valid_samples = load_valid_samples(cache_file)

    if valid_samples is not None:
        print("Loaded valid samples from cache")
    else:
        #make cache file save
        valid_samples = []
        for path, label in dataset.samples:
            if tryloader(path) is not None:
                valid_samples.append((path, label))
        save_valid_samples(valid_samples, cache_file)
        print("Filtered valid samples and saved to cache")

    #ensure only valid samples remain in the dataset
    dataset.samples = valid_samples
    dataset.targets = [label for _, label in valid_samples]
    return dataset



def get_dataset():
    dataset = setup()
    balanced_dataset, org_count = balance_classes(dataset)
    
    sample = lambda batch: augi(batch)
    
    dataloader = DataLoader(balanced_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True, collate_fn=sample)

    return dataset, balanced_dataset, org_count, dataloader
