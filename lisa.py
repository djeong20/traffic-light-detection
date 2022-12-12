import numpy as np
import cv2
import torch

class LisaDataset(torch.utils.data.Dataset):
    def __init__(self, df, transforms=None):
        super().__init__()

        # Image_ids will be the "Filename" here
        self.image_ids = df.image_id.unique()
        self.df = df
        self.transforms = transforms

    def load_image(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0

        return image
        
    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        data = self.df[self.df.image_id == image_id]

        # Load Image
        image = self.load_image(index)
        
        # Bounding Boxes
        bboxes = data[['x_min','y_min','x_max','y_max']].values
        bboxes = torch.as_tensor(bboxes,dtype=torch.float32)
        
        # Box Area (x_max - x_min) * (y_max - y_min)
        # area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        # area = torch.as_tensor(area, dtype=torch.float32)

        # Labels
        labels = torch.as_tensor(data.label.values, dtype=torch.int64)
        # iscrowd = torch.zeros_like(labels, dtype=torch.int64)
        
        target = {'image_id': torch.tensor([index]), 'boxes': bboxes, 'labels': labels}

        if self.transforms:
            sample = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
            sample = self.transforms(**sample)
            
            image = sample['image']
            target['boxes'] = torch.as_tensor(sample['bboxes'],dtype=torch.float32)
            target['labels'] = torch.as_tensor(sample['labels'])
            
        return image, target, image_id

