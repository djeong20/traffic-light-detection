# traffic-light-detection

## Getting Started
### Dependency
- requirements : numpy, pandas, matplotlib, pytorch, opencv-python, torchvision, albumentations
```
pip3 install numpy pandas matplotlib opencv-python torch torchvision albumentations
```

### Download LISA dataset for training from [here](https://www.kaggle.com/datasets/mbornoe/lisa-traffic-light-dataset/download?datasetVersionNumber=2)

### Folder structure

The structure of folder as below.

```
traffic-light-detection
├── lisa.py
├── loader.py
├── train.py
├── demo.py
└── dataset
    ├── Annotations
    ├── dayTrain
    ├── nightTrain
    └── ...
```

### Run demo with pretrained model

```
 python3 demo.py \
--saved_model traffic_light_detector.pth
--nsm_th 0.2 \
--score_th 0.4
```

### Training and evaluation

```
 python3 demo.py \
--batch_size 16 \
--epoch 5 \
--s 0.5 \
--workers 4 \
--grad_clip 5
```

### Arguments
* `--saved_model`: path to saved_model to evaluation.
* `--nms_th`: non maximum suppression threshold
* `--scroe_th`: score threshold
* `--batch_size`: input batch size
* `--epoch`: number of epoches
* `--s`: lr scheduler step size
* `--workers`: number of data loading workers
* `--grad_clip`: gradient clipping value
