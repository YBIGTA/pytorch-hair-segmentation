# pytorch-hair-segmentation
Implementation of pytorch semantic segmentation with [figaro-1k](http://projects.i-ctm.eu/it/progetto/figaro-1k).

### Downloading dataset
```bash
# specify a directory for dataset to be downloaded into, else default is ./data/
sh data/figaro.sh #<directory>
```
### Running trainer

```bash
# sample execution

python3 main.py \
  --networks segnet \
  --dataset figaro \
  --data_dir ./data/Figaro1k \
  --scheduler ReduceLROnPlateau \
  --batch_size 4 \
  --epochs 100 \
  --lr 1e-3 \
  --num_workers 4 \
  --optimizer adam \
  --img_size 256 \
  --momentum 0.5
```

* You should add your own model script in `networks` and make it avaliable in  `get_network` in `./networks/__init__.py`

### Current Project Tree

```python
pytorch-hair-segmentation/
  data/ # includes data script
  docker/ # includes dockerfile
  logs/ # log file for train-test
  markdowns/ # documentation
  models/ # saved model
  networks/ # pytorch model
    deeplab_v3_plus
    mobile_hair
    pspnet
    segnet
    ternausnet
    unet
  notebooks/ # notebook example for using network code
  utils/ # util function for training
```

### RUN with Docker
