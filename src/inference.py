from src.dataset import get_datasets
from src.dataset import CarDetectionDataset
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import random
import matplotlib.pyplot as plt
import numpy as np
from torchvision.ops import nms
import cv2


def inference(model, valset):
    # Load the model's state dictionary from a saved file
    model.load_state_dict(
        torch.load('../trained_models/fasterrcnn_resnet50_fpn_fine_tuned.pth', map_location=torch.device('cpu'))
    )

    # Get a random index for visualization from the validation set
    idx = random.randint(1, len(valset) - 1)
    image, target = valset[idx]

    # Set the model to evaluation mode
    model.eval()

    # Make predictions on the input image
    pred = model([image])

    # Convert the image to NumPy for visualization
    image = image.permute(1, 2, 0).detach().cpu().numpy().copy()

    # Extract bounding boxes and scores from the prediction
    bbox = pred[0]['boxes'].detach().cpu().numpy()
    score = pred[0]['scores'].detach().cpu().numpy()

    # Set a score threshold for filtering predictions
    score_threshold = np.where(pred[0]['scores'].detach().cpu().numpy() > 0.4)[0]

    # Apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes
    mns = nms(torch.Tensor(bbox), torch.Tensor(score), 0.05)
    filtering = np.intersect1d(mns, score_threshold)

    # Filter bounding boxes and scores based on NMS and score threshold
    bbox = bbox[filtering]
    score = score[filtering]

    # Visualize the bounding boxes on the image
    for i in range(len(bbox)):
        sp = (int((bbox[i][0]).item()), int((bbox[i][1]).item()))
        ep = (int((bbox[i][2]).item()), int((bbox[i][3]).item()))
        cv2.rectangle(image, sp, ep, (0, 255, 0), 1)
        cv2.putText(image, str(score[i])[:4], (sp[0], sp[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # Display the image with bounding boxes and scores
    plt.imshow(image)
    plt.axis('off')
    plt.title(str(len(filtering)))
    plt.show()

if __name__ == "__main__":

    # init the validation dataset with the custom class
    _, valset = get_datasets()

    # Init the model with pretrained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    # Running inference
    inference(model, valset)