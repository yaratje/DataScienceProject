""""
All stuff related to data
"""

import torch


""""
Function that splits dataset into train and test datasets
@dataset, the entire dataset
@ratio, ratio beteen 0-1 what the split should be
"""
def train_test_split(dataset, ratio):
    
    # Ex. 2.1a your code here
    train_length = int(len(dataset) * ratio)
    test_length = len(dataset) - train_length

    labels = [dataset.targets[i] for i in range(len(dataset))]
    img =  [img for img, data in dataset]
    training_data_x, testing_data_x = torch.utils.data.random_split(img, [train_length, test_length])
    training_data_y, testing_data_y = torch.utils.data.random_split(labels, [train_length, test_length])
    return training_data_x, testing_data_x, training_data_y, testing_data_y