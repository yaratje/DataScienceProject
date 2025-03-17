import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score

def cal_sil_score(data_mean):
    
    X = np.array(list(data_mean.values()))  # Convert means to matrix
    y = np.array(list(data_mean.keys()))
    # Generate sample data with ground-truth labels
    X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=4,
        cluster_std=1,
        center_box=(-10.0, 10.0),
        shuffle=True,
        random_state=1,
    )  # For reproducibility

    # Compute silhouette scores for the ground-truth labels
    silhouette_avg = silhouette_score(X, y)
    print(f"Overall Silhouette Score: {silhouette_avg:.4f}")

    # Compute silhouette values for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Silhouette plot
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (len(np.unique(y)) + 1) * 10])

    y_lower = 10
    for i, label in enumerate(np.unique(y)):
        # Get silhouette scores for samples of this label
        ith_cluster_silhouette_values = sample_silhouette_values[y == label]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / len(np.unique(y)))
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
        y_lower = y_upper + 10  # Add gap

    ax1.set_title("Silhouette plot for ground-truth labels")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Scatter plot of the dataset with ground-truth labels
    colors = cm.nipy_spectral(y.astype(float) / len(np.unique(y)))
    ax2.scatter(X[:, 0], X[:, 1], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k")

    ax2.set_title("Visualization of data with ground-truth labels")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis using ground-truth labels",
        fontsize=14,
        fontweight="bold",
    )

    plt.save("Images/silscore.png")
