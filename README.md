# pytorch-hair-segmentation
Implementation of pytorch semantic segmentation with [figaro-1k](http://projects.i-ctm.eu/it/progetto/figaro-1k).

- tutorial document : https://pytorchhair.gitbook.io/project/ (kor)

### Prerequisites
```
opencv-contrib-python 3.4.4
pytorch 0.4.1
torchvision 0.2.1
numpy 1.14.5
git-lfs 2.3.4 (to download uploaded model files)
```


### Downloading dataset
```bash
# specify a directory for dataset to be downloaded into, else default is ./data/
sh data/figaro.sh #<directory>
```
### Running trainer

```bash
# sample execution

python3 main.py \
  --networks mobilenet \
  --dataset figaro \
  --data_dir ./data/Figaro1k \
  --scheduler ReduceLROnPlateau \
  --batch_size 4 \
  --epochs 5 \
  --lr 1e-3 \
  --num_workers 2 \
  --optimizer adam \
  --img_size 256 \
  --momentum 0.5 \
  --ignite True
```

* You should add your own model script in `networks` and make it avaliable in  `get_network` in `./networks/__init__.py`

### Running docker & train

> with ignite

`docker run davinnovation/pytorch-hairsegment:cpu python main.py`

> with no-ignite

`docker run -p davinnovation/pytorch-hairsegment:cpu python main.py --ignite False`

### Evaluating model

```bash
# sample execution

python3 evaluate.py \
  --networks pspnet_resnet101 \
  --ckpt_dir ./models/pspnet_resnet101_sgd_lr_0.002_epoch_100_test_iou_0.918.pth \
  --dataset figaro \
  --data_dir ./data/Figaro1k \
  --save_dir ./overlay/ \
  --use_gpu True
```

### Evaluation result on figaro testset

|       Model      | IoU | F1-score |
|       ---        | --- |    ---   |
| pspnet_resnet101 | 0.92|   0.96   |
| pspnet_squeezenet| 0.88|   0.91   |
|   deeplabv3plus  | 0.80|   0.89   |


### Sample visualization
* Red: GT / Blue: Segmentation Map

<a href='https://github.com/YBIGTA/pytorch-hair-segmentation/blob/master/assets/imgs/sample_0.png'><img src='assets/imgs/sample_0.png' alt='sample_0' width=512 height=256 /></a>
<a href='https://github.com/YBIGTA/pytorch-hair-segmentation/blob/master/assets/imgs/sample_1.png'><img src='assets/imgs/sample_1.png' alt='sample_1' width=512 height=256 /></a>
<a href='https://github.com/YBIGTA/pytorch-hair-segmentation/blob/master/assets/imgs/sample_2.png'><img src='assets/imgs/sample_2.png' alt='sample_2' width=512 height=256 /></a>
<a href='https://github.com/YBIGTA/pytorch-hair-segmentation/blob/master/assets/imgs/sample_3.png'><img src='assets/imgs/sample_3.png' alt='sample_3' width=512 height=256 /></a>
<a href='https://github.com/YBIGTA/pytorch-hair-segmentation/blob/master/assets/imgs/sample_4.png'><img src='assets/imgs/sample_4.png' alt='sample_4' width=512 height=256 /></a>
