## [Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)


![](https://img-blog.csdnimg.cn/2018110821502579.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTQzODAxNjU=,size_16,color_FFFFFF,t_70)

앞서 FCN에 대해서 살펴보았습니다. FCN 계열의 모델은 괜찮은 성능을 보여왔지만, 다양한 풍경에서의 일반화 능력이 떨어진다는 단점이 있었습니다. 이미지 전체에서 컨텍스트 정보를 뽑아내는 능력이 부족했기 때문인데요. 예를 들어 위 이미지 가장 윗부분에서 기존의 FCN은 보트를 자동차로 탐지합니다. 이미지 전반에 걸쳐서 정보를 뽑아낼 수 있다면, '물 위에 있기 때문에 이 물체는 보트다.'는 추론이 가능하겠죠. 이를 극복하기 위해 여러 노력이 있었는데요, 대표적으로 dilated convolution을 사용하는 방법이 있습니다. 기존의 convolution에서 커널 사이에 일정한 간격을 두어 필터가 보다 넓은 영역의 정보를 담을 수 있게 합니다. 본 논문 Pyramid Scene Parsing Network 역시 dilated convolution을 사용합니다. 이에 덧붙여 보다 직접적으로 이미지 전역에 걸친 feature를 뽑아내는 모듈을 추가했는데요, 이에 대해서는 아래에서 보다 자세히 설명하겠습니다.


![Dilated convolution](https://www.researchgate.net/profile/Yizhou_Yu/publication/316899215/figure/fig7/AS:668532287238147@1536401926298/Dilated-convolution-Top-The-spatial-resolution-of-feature-maps-is-reduced-by-half-after.png)


저자는 논문의 기여를 아래와 같이 서술합니다.
* FCN 계열의 픽셀 단위 예측 모델에 비해 보다 어려운 풍경 정보를 담을 수 있는 pyramid scene parsing network 제안.
* deeply supervised loss를 활용한 효과적인 최적화 전략 개발.
* 최고 성능을 보이는 semantic segmentation 시스템 개발 및 구현 코드 공개


### Pyramid Pooling Module
기존의 FCN과 가장 차별화되는 부분입니다. 일반적인 FCN은 convolutional layer를 통해 인코딩한 정보를 점진적으로 보간(interpolation)해나갑니다. 보간된 정보는 다시 convolutional layer를 거치며 보다 풍부(dense)해집니다. 이런 과정을 거쳐 입력 이미지에 상응하는 사이즈의 score map을 출력합니다. 따라서 각 픽셀별로 어떤 class에 속하는 지 score를 나타내게 됩니다. 본 논문은 이 중간 과정에 Pyramid Pooling Layer를 추가합니다. 그 구조는 아래와 같습니다.
![](https://hszhao.github.io/projects/pspnet/figures/pspnet.png)

1. 서로 다른 사이즈가 되도록 여러 차례 pooling을 합니다. 논문에서는 1x1, 2x2, 3x3, 6x6 사이즈로 만들었는데요. 1x1 사이즈의 feature map은 가장 조악한 정보를, 하지만 가장 넓은 범위의 정보를 담습니다. 각각 다른 사이즈의 feature map은 서로 다른 부분 영역들의 정보를 담게 됩니다. 
2. 이후 1x1 convolution을 통해 channel 수를 조정합니다. pooling layer의 개수를 N이라고 할 때, `출력 channel 수 = 입력 채널 수 / N` 이 됩니다.
3. 이후 이 모듈의 input size에 맞게끔 feature map을 upsample합니다. 이 과정에서 bilinear interpolation이 사용됩니다.
4. 원래의 feature map과 위 과정을 통해 생성한 새로운 feature map들을 이어붙여(concatenate)줍니다.

본 논문에서는 위와 같은 4개의 사이즈를 사용했지만, 구현에 따라서 다르게 설정할 수 있다고 합니다. pooling의 경우 max pooling과 average pooling을 모두 사용해본 결과, average pooling이 일반적으로 좋은 성능을 보였다고 합니다. 

### deeply supervised loss
![](https://tangzhenyu.github.io/assets/paper_notes/pspnet/image3.jpg)
본 논문에서는 dilated convolution을 사용한 resnet 50, resnet101 등 깊은 모델을 사용했습니다. 깊은 모델이 잘 학습될 경우 더 좋은 정확도를 보이지만, 최적화가 어려운데요. 본 논문에서는 보조적인 loss를 사용하여 이 문제를 해결합니다. 모델을 끝단에서 loss1을 산출하기에 앞서, 앞단의 레이어 (resnet4b22)에 보조적인 classifier를 달아 보조 loss2를 산출합니다. 트레이닝 과정에서 이 두 가지 loss를 가중합계 내어 최종적인 loss를 산출합니다. Inference단에서는 loss2를 계산하는 보조 classifier는 사용하지 않는다고 합니다. 본 논문에선는 여러 실험을 통해 이 보조 loss의 존재가 학습에 도움이 된다고 주장합니다.


### 구현
설명에 앞서 기존의 [pytorch pspnet 구현체](https://github.com/Lextal/pspnet-pytorch)를 참고했음을 밝힙니다.

1. Base Network 
 - 저희가 사용하는 Figaro-1k은 데이터셋 크기가 작기 때문에 최대한 얕은 모델을 사용하고자 했습니다.
 - 때문에 SqueezeNet을 사용했습니다. `torchvision.models`에서 pretrained model을 불러와 classifier 전단계까지의 레이어를 사용했습니다.

 ```python
 class SqueezeNetExtractor(nn.Module):
    def __init__(self):
        super(SqueezeNetExtractor, self).__init__()
        model = squeezenet1_1(pretrained=True)
        features = model.features
        self.feature1 = features[:2]
        self.feature2 = features[2:5]
        self.feature3 = features[5:8]
        self.feature4 = features[8:]

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)
        return f4
 ```
2. Pyramid Pooling Layer
    - 논문에 나온 4 가지 사이즈 (1x1, 2x2, 3x3, 6x6)를 사용했습니다.
    - 이 과정에서는 `torch.nn.AdaptiveAvgPool2d`를 사용했습니다.
    ```python
    class PyramidPoolingModule(nn.Module):
        def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
            super(PyramidPoolingModule, self).__init__()
            pyramid_levels = len(sizes)
            out_channels = in_channels // pyramid_levels

            pooling_layers = nn.ModuleList()
            for size in sizes:
                layers = [nn.AdaptiveAvgPool2d(size), nn.Conv2d(in_channels, out_channels, kernel_size=1)]
                pyramid_layer = nn.Sequential(*layers)
                pooling_layers.append(pyramid_layer)

            self.pooling_layers = pooling_layers

        def forward(self, x):
            h, w = x.size(2), x.size(3)
            features = [x]
            for pooling_layer in self.pooling_layers:
                # pool with different sizes
                pooled = pooling_layer(x)

                # upsample to original size
                upsampled = F.upsample(pooled, size=(h, w), mode='bilinear')

                features.append(upsampled)

            return torch.cat(features, dim=1)
    ```
3. deeply supervised loss
 - SqueezeNet을 사용했기 때문에 보조적인 loss가 필요 없다고 판단하여 구현하지 않았습니다.


### 성능 평가

pyramid pooling 모듈의 사용 여부가 정확도에 미치는 영향을 파악하기 위해 pyramid pooling 모듈을 제외한 FCN 형태의 네트워크도 학습을 진행해보았습니다.

1. 양적 평가

| pyramid pooling | threshold | Validation-loss | Pixcel Accuracy |  IoU  | F1-score|
|      ---        |    ---    |       ---       |      ---        |  ---  |   ---   |
|       o         |    0.50   |      0.11       |      0.96       | 0.861 |  0.925  |
|       x         |    0.50   |      0.12       |      0.95       | 0.843 |  0.915  |

* Pixel Accuracy = TP + TN / (TP + FP + FN + TN)
* IoU = TP / (TP + FP + FN)
* F1-score = 2 * precision * recall / () precision + recall )
* 위 결과는 각각의 네트워크에서 가장 IoU가 가장 높은 pyramid pooling module을 사용하는 경우 대부분의 measure에서 보다 좋은 정확도를 보였습니다.


2. 질적 평가

* Best Cases

<a href="http://www.freeimagehosting.net/commercial-photography/"><img src="https://i.imgur.com/3Re6OwC.png" alt="Commercial Photography" width=512 height=256></a>
<a href="http://www.freeimagehosting.net/commercial-photography/"><img src="https://i.imgur.com/G465OTX.png" alt="Commercial Photography" width=512 height=256></a>
<a href="http://www.freeimagehosting.net/commercial-photography/"><img src="https://i.imgur.com/hjctyLx.png" alt="Commercial Photography" width=512 height=256></a>
<a href='https://postimg.cc/627dPcBv' target='_blank'><img src='https://i.postimg.cc/627dPcBv/SAMPLE-5.png' alt='SAMPLE-5' width=512 height=256 /></a>

* Failure Cases

<a href="http://www.freeimagehosting.net/commercial-photography/"><img src="https://i.imgur.com/txjqgS2.jpg" alt="Commercial Photography" width=512 height=256></a>
<a href="http://www.freeimagehosting.net/commercial-photography/"><img src="https://i.imgur.com/YALQvCk.jpg" alt="Commercial Photography" width=512 height=256></a>
    * 흑백 이미지에서 머리카락이 아닌 다른 부분이 탐지되는 경우가 있습니다.
    * 수염 등 머리카락과 헷갈릴만한 부분에서 잘못 탐지되는 경우가 있습니다.
    * 보다 많은 샘플을 확보한다면 일반화 능력이 더 좋아질 것으로 예상됩니다.




