from data_loading import get_dataset as load_data
from data_understanding import *


def main():
    dataset, balanced_dataset, org_count, dataloader = load_data()
    class_counts(balanced_dataset, org_count, dataset.classes)
    intra_corr(dataloader)
    four_images(dataloader, sorted(set(dataset.targets)))


def get_dataloader():
    _,_,_, dataloader = load_data()
    return dataloader

if __name__ == '__main__':
    main()

