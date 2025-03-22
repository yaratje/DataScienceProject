
from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import matplotlib.cm as cm
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

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
Visualization code adapted from: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
Takes dataloader and uses intra_corr method to calculate data_mean which is used for the silloute score.
""" 
def cal_sil_score(dataset):
    # Compute feature vectors (means) for each sample in the dataset
    data_mean, _ = intra_corr(dataset)
    
    # Combine the features from all classes into X and y arrays
    X, y = [], []
    for label, means in data_mean.items():
        for feat in means:
            X.append(feat)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    
    # Compute overall silhouette score
    silhouette_avg = silhouette_score(X, y)
    print(f"Overall Silhouette Score: {silhouette_avg:.4f}")
    
    # Compute silhouette values for each sample
    sample_silhouette_values = silhouette_samples(X, y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Silhouette plot setup
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(X) + (len(np.unique(y)) + 1) * 10])
    
    y_lower = 10
    for i, label in enumerate(np.unique(y)):
        ith_silhouette_vals = sample_silhouette_values[y == label]
        ith_silhouette_vals.sort()
        size_cluster_i = ith_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        color = cm.nipy_spectral(float(i) / len(np.unique(y)))
        ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_silhouette_vals,facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10  # gap between clusters

    ax1.set_title("Silhouette Plot for Dataset")
    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster Label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # Scatter plot of the dataset features colored by class label
    colors = cm.nipy_spectral(y.astype(float) / len(np.unique(y)))
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")
    ax2.set_title("Dataset Visualization with Ground-Truth Labels")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    
    plt.suptitle("Silhouette Analysis of Dataset", fontsize=14, fontweight="bold")
    plt.savefig("Images/silscore.png")

