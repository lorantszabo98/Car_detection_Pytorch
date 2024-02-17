from src.dataset import CarDetectionDataset
from src.model import create_model
from src.train import train_model
import pandas as pd
from sklearn.model_selection import train_test_split
import albumentations as A
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from torchvision.ops import nms
import cv2
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


image_size = 256

# Augs
train_augs=A.Compose([
    A.Resize(image_size,image_size),
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.2),
    A.RandomBrightnessContrast(p=0.2),
],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), is_check_shapes=False)

val_augs=A.Compose([
    A.Resize(image_size,image_size),
],bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']), is_check_shapes=False)

test_augs=A.Compose([
    A.Resize(image_size,image_size)
],is_check_shapes=False)


# Load your annotations into a DataFrame
df = pd.read_csv('data/train_solution_bounding_boxes (1).csv') # Update the path
df['label'] = 1

# Split your data into train and validation sets
train_img_ids, val_img_ids = train_test_split(df.image.unique(), test_size=0.1, random_state=32)

train_df = df[df.image.isin(train_img_ids)]
val_df = df[df.image.isin(val_img_ids)]

# Create instances of the dataset for training and validation
trainset = CarDetectionDataset(train_img_ids, train_df, augs=train_augs)
valset = CarDetectionDataset(val_img_ids, val_df,augs=val_augs)


# Define the model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

# Train the model
# train_model(model, trainset, num_epochs=10)
#
# # Evaluate the model on the test set
# evaluate_model(model, test_dataset)


# Evaluation
model.load_state_dict(torch.load('trained_models/fasterrcnn_resnet50_fpn_fine_tuned.pth', map_location=torch.device('cpu')))
idx=random.randint(1,len(valset)-1)
image, target = valset[idx]
image = image.permute(1,2,0).numpy()
image = test_augs(image=image)['image']

plt.imshow(image)

img = torch.Tensor(np.transpose(image,(2,0,1)))

model.eval()
pred = model([img])

image = img.permute(1,2,0).detach().cpu().numpy().copy()
bbox = pred[0]['boxes'].detach().cpu().numpy()
score = pred[0]['scores'].detach().cpu().numpy()

score_threshold = np.where(pred[0]['scores'].detach().cpu().numpy()>0.4)[0]
ixs = nms(torch.Tensor(bbox),torch.Tensor(score),0.05)
filt = np.intersect1d(ixs, score_threshold)

bbox = bbox[filt]
score = score[filt]

for i in range(len(bbox)):
    sp=(int((bbox[i][0]).item()),int((bbox[i][1]).item()))
    ep=(int((bbox[i][2]).item()),int((bbox[i][3]).item()))
    cv2.rectangle(image, sp, ep, (0,255,0), 1)
    cv2.putText(image, str(score[i])[:4], (sp[0], sp[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1)

plt.imshow(image)
plt.axis('off')
plt.title(str(len(filt)))
plt.show()

