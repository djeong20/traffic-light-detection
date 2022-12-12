from loader import LISA
from lisa import LisaDataset
from utils import Averager

import argparse
import tqdm

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def collate_fn(batch):
    return tuple(zip(*batch))

def train(args):
    # Load Dataset
    lisa = LISA()
    df = lisa.load_dataset('dataset/Annotations/Annotations/dayTrain', 'dataset/Annotations/Annotations/nightTrain')
    train_df, test_df = lisa.train_test_split(df, 0.2)
    train_dataset = LisaDataset(train_df, A.Compose([A.Resize(height=512, width=512, p=1), ToTensorV2(p=1.0)], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']}))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers= args.workers, collate_fn=collate_fn)

    # Model: Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(input_features, 4)

    params = [p for p in model.parameters() if p.requires_grad]

    # Optimizers
    optimizer = torch.optim.Adam(params)

    # LR Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = args.step_size)
    loss_avg = Averager()

    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device =  torch.device('cpu')

    model.to(device)
    
    """ start training """
    for epoch in range(args.n_epochs):
        # train part
        model.train()
        loss_avg.reset()
        
        for (imgs, targets, image_ids) in tqdm.tqdm(train_dataloader):
            imgs = torch.stack(imgs).to(device)

            y = []
            for target in targets:
                data = dict()
                for key, val in target.items():
                    data[key] = val.to(device)
                y.append(data)
            
            preds = model(imgs, y)
            
            cost = sum(loss for loss in preds.values())
            loss_avg.update(cost.item(), imgs.shape[0])

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # gradient clipping with 5 (Default)
            optimizer.step()
        
        scheduler.step(cost)

        print(f"Epoch {epoch+1}/{args.n_epochs}")
        print(f"Train loss: {loss_avg.val():0.5f}")
        loss_avg.reset()
        
        torch.save(model.state_dict(), 'traffic_light_detector.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', dest="batch_size", type=int, default=16, help='input batch size')
    parser.add_argument('--epoch', dest="n_epochs", type=int, default=5, help='number of epoch')
    parser.add_argument('--s', dest="step_size", type=float, default=0.5, help='step size of lr scheduler')
    parser.add_argument('--workers', dest="workers", type=int, default=4, help='number of data loading workers')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    
    opt = parser.parse_args()
    train(opt)