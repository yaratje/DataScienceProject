
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import cv2


"""
Calculate the intra corralation of both the mean and the standard deviation, save as csv.
"""
def intra_corr(dataloader):
    data_mean = {}
    data_std = {}
    
    for imgs, labels in dataloader:
        for img, label in zip(imgs, labels):
            img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.float32)
            mean, std = cv2.meanStdDev(img_np)
            #mean feature per label, set label if not in dict yet
            label_val = label.item()
            data_mean.setdefault(label_val, []).append(np.squeeze(mean))  
            data_std.setdefault(label_val,[]).append(np.squeeze(std))
    df_class_means = pd.DataFrame({label: np.mean(means, axis=0) for label, means in data_mean.items()})
    df_class_std = pd.DataFrame({label: np.mean(std, axis=0) for label, std in data_std.items()})


    #save correlation matrix
    df_class_means.corr().to_csv("Results/correlation_matrix_mean.csv")
    df_class_std.corr().to_csv("Results/correlation_matrix_std.csv")
    return data_mean, data_std



"""
Get 4 images per class, display them, result saved to Images folder
"""
def four_images(dataloader, unique_labels):

    samples = {cls: [] for cls in unique_labels}

    #collect all 4 imagages for each label
    for imgs, labels in dataloader:
        for img, label in zip(imgs, labels):
            label = int(label.item())
            if len(samples[label]) < 4:
                samples[label].append(img)
            if all(len(img_list) == 4 for img_list in samples.values()):
                break 

    fig, axes = plt.subplots(len(unique_labels), 4, squeeze=False, figsize=(12, 3*len(unique_labels)))
    for row, cls in enumerate(unique_labels):
            for col in range(4):
                ax = axes[row, col]
                # Transpose image dimensions, denormalizing it for the sake of displaying, maybe should do this differently :()
                img = samples[cls][col].numpy()
                ax.imshow(np.transpose(img*0.5 + 0.5, (1, 2, 0)))
                ax.set_xticks([])
                ax.set_yticks([])
            axes[row, 0].set_ylabel(cls, rotation=0, labelpad=50, fontsize=12, ha="right", va="center")
    plt.suptitle("Sample Images from Each Class", fontsize=16, fontweight="bold")
    plt.savefig("Images/4img.jpg")


"""
Count the ammount of images per class, and compare before and after data augmentation.(doesnt need the actual images, so use balance list instead of dataloader)
"""

def class_counts(balanced_list, org_count, class_names):
    #count images per class
    balanced_labels = [label for (path, label, flag) in balanced_list]
    balanced_count = Counter(balanced_labels)
    
    unique_labels = sorted(balanced_count.keys())
    bar_width = 0.4  
    x = np.arange(len(unique_labels))
    
    plt.figure(figsize=(10, 6))
    #balanced dataset counts
    plt.bar(x - bar_width/2, [balanced_count[label] for label in unique_labels],
            width=bar_width, label="Balanced", color='b')
    #original counts
    plt.bar(x + bar_width/2, [org_count[label] for label in unique_labels],
            width=bar_width, label="Original", color='r')
    
    plt.title('Number of Samples per Category')

    plt.xticks(ticks=x, labels=[class_names[label] for label in unique_labels], rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Images/class_count_barplot_compare.jpg")



""""
Function that splits dataset into train and test datasets
@dataset, the entire dataset
@ratio, ratio beteen 0-1 what the split should be (for this project set to 0.2/0.8 test/train)
"""
def split_data(ratio, dataset):
    labels = np.array(dataset.targets)
    #stratify=labels = balanced labels distribution :O
    train_data_x,test_data_x,train_data_y,test_data_y = train_test_split(dataset.samples,labels, test_size=ratio, stratify=labels)

    return train_data_x,test_data_x,train_data_y,test_data_y
    
"""
Count the amount of images per label, and create a bar plot, result saved to Images folder
"""