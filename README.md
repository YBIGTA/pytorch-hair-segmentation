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

`docker run -p :8097 davinnovation/pytorch-hairsegment:cpu python main.py --ignite False`
