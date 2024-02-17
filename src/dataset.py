import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np


# Torch Dataset
class CarDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_ids, df, augs=None, train=True):
        self.df = df
        self.augs = augs
        self.image_ids = image_ids
        if train:
            self.dir_path = './data/training_images'
        else:
            self.dir_path = './data/testing_images'

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