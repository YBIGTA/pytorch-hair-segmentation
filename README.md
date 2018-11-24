# pytorch-hair-segmentation
Implementation of pytorch semantic segmentation with [figaro-1k](http://projects.i-ctm.eu/it/progetto/figaro-1k).

### Downloading dataset
```bash
# specify a directory for dataset to be downloaded into, else default is ./data/
sh data/figaro.sh #<directory>
```
### Running trainer

```bash
# run this in root

python3 main.py \
  --networks segnet \
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