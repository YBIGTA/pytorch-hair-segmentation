# pytorch-hair-segmentation
Implementation of pytorch semantic segmentation with [figaro-1k](http://projects.i-ctm.eu/it/progetto/figaro-1k).

- tutorial document : https://pytorchhair.gitbook.io/project/ (kor)

### Prerequisites
```
opencv-contrib-python 3.4.4
pytorch 0.4.1
torchvision 0.2.1
numpy 1.14.5
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
  --ckpt_dir [path to checkpoint] \
  --dataset figaro \
  --data_dir ./data/Figaro1k \
  --save_dir ./overlay/ \
  --use_gpu True
```

### Evaluation result on figaro testset

|       Model      | IoU | F1-score |      Checkpoint      |
|       ---        | --- |    ---   |          ---         |
| pspnet_resnet101 | 0.92|   0.96   | [link](https://drive.google.com/file/d/1w7oMuxckqEClImjLFTH7xBCpm1wg7Eg4/view?usp=sharing)
| pspnet_squeezenet| 0.88|   0.91   | [link](https://drive.google.com/file/d/1ieKvsK3uoDZN0vA5MenQphca4AZZuaPk/view?usp=sharing) |
|   deeplabv3plus  | 0.80|   0.89   |   - |


### Sample visualization
* Red: GT / Blue: Segmentation Map

![sample_0](https://user-images.githubusercontent.com/19547969/227229779-28b42d02-efad-4b7b-be65-3cf3a1a7bfef.png)
![sample_1](https://user-images.githubusercontent.com/19547969/227229796-5de39ea1-73fe-4be8-9ef7-2857df54b94c.png)
![sample_2](https://user-images.githubusercontent.com/19547969/227229856-e224b91c-6fb2-4aa8-a93f-6ab1edfe568b.png)
![sample_3](https://user-images.githubusercontent.com/19547969/227229883-ff4b05e7-ba23-42c9-9dec-a431bf0715f1.png)
![sample_4](https://user-images.githubusercontent.com/19547969/227229909-68b6cdf1-6f89-4cf9-a2f8-01be251dd140.png)
