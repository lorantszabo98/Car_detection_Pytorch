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
    model.load_state_dict(
        torch.load('../trained_models/fasterrcnn_resnet50_fpn_fine_tuned.pth', map_location=torch.device('cpu')))
    idx = random.randint(1, len(valset) - 1)
    image, target = valset[idx]
    image = image.permute(1, 2, 0).numpy()
    # image = test_augs(image=image)['image']

    plt.imshow(image)

    img = torch.Tensor(np.transpose(image, (2, 0, 1)))

    model.eval()
    pred = model([img])

    image = img.permute(1, 2, 0).detach().cpu().numpy().copy()
    bbox = pred[0]['boxes'].detach().cpu().numpy()
    score = pred[0]['scores'].detach().cpu().numpy()

    score_threshold = np.where(pred[0]['scores'].detach().cpu().numpy() > 0.4)[0]
    ixs = nms(torch.Tensor(bbox), torch.Tensor(score), 0.05)
    filt = np.intersect1d(ixs, score_threshold)

    bbox = bbox[filt]
    score = score[filt]

    for i in range(len(bbox)):
        sp = (int((bbox[i][0]).item()), int((bbox[i][1]).item()))
        ep = (int((bbox[i][2]).item()), int((bbox[i][3]).item()))
        cv2.rectangle(image, sp, ep, (0, 255, 0), 1)
        cv2.putText(image, str(score[i])[:4], (sp[0], sp[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    plt.imshow(image)
    plt.axis('off')
    plt.title(str(len(filt)))
    plt.show()

if __name__ == "__main__":

    _, valset = get_datasets()

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(preTrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    inference(model, valset)