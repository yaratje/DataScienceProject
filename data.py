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
import json, os
import matplotlib.pyplot as plt


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
def preprocess_tensor(dataset, cache_path="cached_images.pt"):
    if os.path.exists(cache_path):
        print("Loading images from cache")
        return torch.load(cache_path)
    else:
        print("Preprocessing images and chacing")
        image_list = []
        for img, _ in dataset:
            image_list.append(img)
        # Stack the images into a single tensor
        images_tensor = torch.stack(image_list)
        torch.save(images_tensor, cache_path)
        return images_tensor

""""
Setup basic variables, get dataset from local folder, create dataloader, and extract labels and images for later use
Additionally, added caching of all the valid file paths as a json file cause filtering the dataset took long, so if file exists, load from there
"""
def setup():
    #TRANSFORM from assignment 2
    global dataset, dataloader, labels, images

    transform = transforms.Compose([transforms.Resize(32), transforms.CenterCrop(32), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    dataset = ImageFolder(root="./Incidents-subset", transform=transform, loader=tryloader)

    cache_file = "valid_samples.json"
    valid_samples = []

    cached = load_valid_samples(cache_file)
    if cached is not None:
        valid_samples = cached
        print("Loaded valid samples from cache")
    else:
        #make cache file save
        for path, label in dataset.samples:
            if tryloader(path) is not None:
                valid_samples.append((path, label))
        save_valid_samples(valid_samples, cache_file)
        print("Filtered valid samples and saved to cache")


    #ensure only valid samples remain in the dataset

    dataset.samples = valid_samples
    dataset.targets = [label for _, label in valid_samples]
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    labels = np.array(dataset.targets)
    
    images_tensor = preprocess_tensor(dataset, cache_path="cached_images.pt")
    #feel like i need em in np array
    images = images_tensor.numpy()


""""
Function that splits dataset into train and test datasets
@dataset, the entire dataset
@ratio, ratio beteen 0-1 what the split should be (for this project set to 0.2/0.8 test/train)
"""
def split_data(dataset, ratio):
    #stratify=labels = balanced labels distribution :O
    train_data_x,test_data_x,train_data_y,test_data_y = train_test_split(images,labels, test_size=ratio, stratify=labels)

    return train_data_x,test_data_x,train_data_y,test_data_y
    
"""
Count the amount of images per label, and create a bar plot, result saved to Images folder
"""
def class_counts():

    class_counts = Counter(labels)

    #print the outcome
    for class_idx, count in class_counts.items():
        print(f"{dataset.classes[class_idx]}: number of images {count}, class index: {class_idx}")

    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('number of samples per category')
    #stupid xaxis labels 
    plt.xticks(ticks=range(len(dataset.classes)), labels=dataset.classes, rotation=90)
    plt.xlabel('classes')
    plt.ylabel('number of samples')
    plt.tight_layout()
    plt.savefig("Images/class_count_barplot.jpg")


if __name__ == '__main__':
    setup()
    class_counts()
    train_data_x, test_data_x , train_data_y, test_data_y = split_data(dataset, 0.2)