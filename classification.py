from data_main import get_dataloader
import torch.nn as nn
import torch
from torchvision import models
from validation import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import time

#use this for training :) augmented dataset with balanced classes, see data_understanding for how to use
dataloader_train, dataloader_test = get_dataloader()


model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

#put le model in the evaluation modee
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def get_features(dataloader):
    features = []
    labels = []
    with torch.no_grad():  # No need to compute gradients
        for images, targets in dataloader:
            images = images.to(device)
            output = model(images)
            # go from (batch_size, 512, 1, 1) to (batch_size, 512), and move output to cpu maybe for easy use later
            output = output.view(output.size(0), -1).cpu()
            features.append(output)
            labels.append(targets.cpu())

    #make into a single tensor to get overall descriptor
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0) 
    return features.numpy(), labels.numpy()

train_CNN, train_data_y = get_features(dataloader_train)
test_CNN, test_data_y = get_features(dataloader_test)

kvalue_list = [2, 4, 6, 10, 15]
kvalue_list2 = [10]

for i in kvalue_list2:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_CNN, train_data_y)

    t0 = time.time()
    predicted = knn.predict(test_CNN)
    predicted2 = knn.predict(train_CNN)
    t1 = time.time()
    print(t1-t0)
    predicted = np.array(predicted, dtype=int)
    predicted2 = np.array(predicted2, dtype=int)
    test_data_y = np.array(test_data_y, dtype=int)
    train_data_y = np.array(train_data_y, dtype=int)

    accuracy_value, recall, precision, fscore = accuracy_metric(test_data_y, predicted)
    #accuracy_value, recall, precision, fscore = accuracy_metric(train_data_y, predicted2)
    print(f"k={i}: Accuracy={accuracy_value}, Recall={recall}, Precision={precision}, F1-score={fscore}")

kfold(knn, train_CNN, train_data_y)
confusion(test_data_y, predicted)


