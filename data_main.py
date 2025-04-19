from data_loading import get_dataset as load_data
from data_understanding import *


def main():
    dataset, balanced_dataset, org_count, dataloader_train, dataloader_test = load_data()
    class_counts(balanced_dataset, org_count, dataset.classes)
    #intra_corr(dataloader_train)
    #four_images(dataloader_train, sorted(set(dataset.targets)))
    #cal_sil_score(dataloader_train)
    #PCA2D(dataloader_train)
    #PCA3D(dataloader_test)


def get_dataloader():
    _,_,_, dataloader_train, dataloader_test = load_data()
    return dataloader_train, dataloader_test

    

if __name__ == '__main__':
    main()

