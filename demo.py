import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse

import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from loader import LISA
from lisa import LisaDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def predict(image_id, output, nms_threshold, score_threshold):
    def box_filter(output, nms_th, score_threshold):
        boxes = output['boxes']
        scores = output['scores']
        labels = output['labels']
        
        # non maximum suppression
        mask = torchvision.ops.nms(boxes, scores, nms_th)
        
        boxes = boxes[mask].data.cpu().numpy().astype(np.int32)
        scores = scores[mask].data.cpu().numpy()
        labels = labels[mask].data.cpu().numpy()

        mask = (scores >= score_threshold)
        
        return boxes[mask], scores[mask], labels[mask]

    boxes, scores, labels = box_filter(output, nms_threshold, score_threshold)
    
    # Red, Yellow, Green
    colors = dict()
    colors[1] = (0, 255, 0)
    colors[2] = (255, 255, 0)
    colors[3] = (255, 0, 0)

    # Preprocessing
    image = cv2.imread(image_id)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = cv2.resize(image,(512,512))
    image /= 255.0
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Draw box in image
    for pos, label in zip(boxes, labels):
        image = cv2.rectangle(image, (pos[0], pos[1]), (pos[2], pos[3]), colors[label], 2)

    # Display image
    ax.set_axis_off()
    ax.imshow(image)
    ax.set_title(image_id)
    plt.show()

def load_dataset():
    print("Loading Dataset...")
    lisa = LISA()
    df = lisa.load_dataset('dataset/Annotations/Annotations/dayTrain', 'dataset/Annotations/Annotations/nightTrain')

    _, test_df = lisa.train_test_split(df, 0.2)
    test_dataset = LisaDataset(test_df, A.Compose([A.Resize(height=512, width=512, p=1), ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    return test_dataloader

def load_model(model_path):
    print("Loading Model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, 4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def demo(args):
    test_dataloader = load_dataset()
    model = load_model(args.saved_model)
    model.eval()

    imgs, targets, image_ids = next(iter(test_dataloader))
    print("Making prediction...")
    preds = model(torch.stack(imgs))
    predict(image_ids[2], preds[2], args.nms_th, args.score_th)

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--saved_model', help="path to saved_model to evaluation", default='traffic_light_detector.pth')
    parser.add_argument('--nms_th', help="Non Maximum Suppression Threshold", default=0.2)
    parser.add_argument('--score_th', help="Score Threshold", default=0.4)

    args = parser.parse_args()
    demo(args)
    
    