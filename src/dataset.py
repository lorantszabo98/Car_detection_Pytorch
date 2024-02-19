import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import albumentations as A
from sklearn.model_selection import train_test_split

image_size = 256

# Torch Dataset
class CarDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, df, augs=None, train=True):
        self.df = df
        self.augs = augs
        self.image_ids = image_ids
        if train:
            self.dir_path = '../data/training_images'
        else:
            self.dir_path = '../data/testing_images'

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.dir_path, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = self.df[self.df.image == image_id][['xmin', 'ymin', 'xmax', 'ymax']].values
        labels = self.df[self.df.image == image_id][['label']].values

        if self.augs != None:
            data = self.augs(image=image, bboxes=bboxes, class_labels=['None'] * len(bboxes))
            image = data['image']
            bboxes = data['bboxes']

        image = torch.Tensor(np.transpose(image, (2, 0, 1))) / 255.0
        bboxes = torch.Tensor(bboxes).long()
        labels = torch.Tensor(labels).long().squeeze(1)

        target = {}
        target['boxes'] = bboxes
        target['labels'] = labels

        return image, target

    def collate_fn(self, batch):
        return tuple(zip(*batch))


def get_datasets():
    # Augs
    train_augs = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), is_check_shapes=False)

    val_augs = A.Compose([
        A.Resize(image_size, image_size),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), is_check_shapes=False)

    test_augs = A.Compose([
        A.Resize(image_size, image_size)
    ], is_check_shapes=False)

    # Load your annotations into a DataFrame
    df = pd.read_csv('../data/train_solution_bounding_boxes (1).csv')  # Update the path
    df['label'] = 1

    # Split your data into train and validation sets
    train_img_ids, val_img_ids = train_test_split(df.image.unique(), test_size=0.1, random_state=32)

    train_df = df[df.image.isin(train_img_ids)]
    val_df = df[df.image.isin(val_img_ids)]

    # Create instances of the dataset for training and validation
    trainset = CarDetectionDataset(train_img_ids, train_df, augs=train_augs)
    valset = CarDetectionDataset(val_img_ids, val_df, augs=val_augs)

    return trainset, valset