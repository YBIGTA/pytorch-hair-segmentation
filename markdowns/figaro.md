## Figaro 데이터셋 소개

저희가 semantic segmentation을 적용하고자 하는 분야는 hair segmentation입니다. 이미지에서 사람의 머리카락 부분을 탐지하는 작업인데요. 이와 관련된 데이터셋 중 하나가 [Figaro-1k](http://projects.i-ctm.eu/it/progetto/figaro-1k) 입니다. strait, wavy, curly 등 7가지 헤어스타일의 클래스가 있습니다. 각각 클래스당 150개로 총 1050개의 데이터셋입니다. 4:1 비율로 Training 840, Testing 210 개로 나뉘어 있습니다. 샘플 이미지는 아래와 같습니다.

![](http://projects.i-ctm.eu/sites/default/files/Images/207_Michele%20Svanera/database.jpg)

Patch1k도 제공하고 있습니다. 머리카락 부분만 줌인하여 편집한 패치 1050장과 non-hair 패치 1050장이 있습니다. [논문](http://www.eecs.qmul.ac.uk/~urm30/Doc/Publication/2018/IVC2018.pdf)에 따르면 hair에 대한 feature를 학습하기 위한 보조 데이터셋으로 사용 가능하다고 합니다. 저희는 보조 데이터셋을 사용하지 않고 Figaro-1k 데이터만 사용하여 학습했습니다. 또한 헤어스타일 클래스에 상관 없이 머리카락인지 아닌지에 대한 binary classification을 진행하였습니다. 프로젝트의 root에서 아래와 같은 명령어로 데이터셋을 다운 받을 수 있습니다. 

```bash
# 특정 디렉토리에 다운로드를 원하는 경우 argument로 명시. 그렇지 않을 시 ./data/ 에 다운로드
sh data/figaro.sh #<directory>
```

## Pytorch FigaroDataSet 구현
파이토치에서는 torch.utils.data.Dataset을 상속받아 손쉽게 데이터셋을 구현할 수 있으며, `torch.utils.data.dataloader`를 통해 쉽게 불러올 수 있습니다. 데이터셋 구현 시 __init__, __getitem__, __len__ 세 가지 메소드를 오버라이딩하면 됩니다.


#### __ init __
다른 메소드 실해엥 필요한 멤버변수를 설정합니다. Figaro1k/ 폴더의 경로를 root_dir로 넣어주면 이미지와 마스크(GT) 파일들의 경로를 저장합니다. 또한 이미지 혹은 GT 마스크에 적용할 transforms들을 인자로 받아 멤버변수로 저장합니다. transforms에 대해서는 아래서 설명하겠습니다.

```python
class FigaroDataset(Dataset):
    def __init__(self, root_dir, train=True, joint_transforms=None,
                 image_transforms=None, mask_transforms=None, gray_image=False):
        """
        Args:
            root_dir (str): root directory of dataset
            joint_transforms (torchvision.transforms.Compose): tranformation on both data and target
            image_transforms (torchvision.transforms.Compose): tranformation only on data
            mask_transforms (torchvision.transforms.Compose): tranformation only on target
            gray_image (bool): whether to return gray image image or not.
                               If True, returns img, mask, gray.
        """
        mode = 'Training' if train else 'Testing'
        img_dir = os.path.join(root_dir, 'Original', mode)
        mask_dir = os.path.join(root_dir, 'GT', mode)

        self.img_path_list = [os.path.join(img_dir, img) for img in sorted(os.listdir(img_dir))]
        self.mask_path_list = [os.path.join(mask_dir, mask) for mask in sorted(os.listdir(mask_dir))]
        self.joint_transforms = joint_transforms
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms
        self.gray_image = gray_image
```


#### __ getitem __
인덱스를 통해 이미지 / GT 파일에 접근한 후 pytorch tensor 형태로 반환하는 함수입니다. 이미지 / GT 파일을 PIL Image로 읽어온 후 세 가지 transforming 과정을 거칩니다.

1. joint_transforms: 좌우변환 등 geometric한 변경이 필요한 경우 이미지(데이터)와 마스트(타겟)에 모두 적용시킵니다.
2. image_transforms: 색상 변환 등 이미지(데이터)에만 적용되는 변환입니다. 
3. mask_transoforms: 마스크(타겟)에만 적용되는 변환입니다. PIL.Image를 Tensor로 변환하는 데만 사용했습니다. 

FigaroDataSet 인스턴스 생성 시 image / mask tranforms의 경우 `torchvision.transforms`의 클래스들을 argument로 넘겨주었습니다. joint transforms의 경우, [pytorch-semantic-segmentation](https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py)의 구현체를 이용했습니다. 또한 grayscale의 이미지가 필요한 경우 함께 리턴하도록 구현했습니다. 

``` python
    def __getitem__(self,idx):
        img_path = self.img_path_list[idx]
        img = Image.open(img_path)

        if self.gray_image:
            gray = img.convert('LA')

        mask_path = self.mask_path_list[idx]
        mask = Image.open(mask_path)

        if self.joint_transforms is not None:
            img, mask = self.joint_transforms(img, mask)

        if self.image_transforms is not None:
            img = self.image_transforms(img)

        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        
        if self.gray_image:
            return img, mask, gray
        else:
            return img, mask
```

#### __ len __
데이터셋의 크기를 반환하는 함수로, 멤버변수에 저장된 마스크 파일 갯수를 반환시켰습니다.
```python
    def __len__(self):
        return len(self.mask_path_list)
```

