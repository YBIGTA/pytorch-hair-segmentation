## FCN (MobileNetV2)

[Real-time deep hair matting on mobile devices](https://arxiv.org/abs/1712.07168) 의 Figure 2에 나온 Fully Convolutional MobileNet Architecture for Hair Segmentation 네트워크입니다. 구현체는 `networks/fcn_mobilenetv2.py` 의 `FCN` 클라스를 보시면 됩니다.



이 논문에서는 MobileNet을 기반으로 모델을 구성했으나, 본 글에서는 조금 더 최근에 나온 MobileNetV2를 기반으로 모델을 구성했습니다. [MobileNetV2](https://arxiv.org/abs/1801.04381)는(이하 V2) inverted residual과 linear bottleneck의 도입으로 기존의 MobileNet보다도 파라미터 수가 적은데, 논문에 따르면 성능은 기존과 비슷하다고 합니다.



### 1. Architecture

![alt Fully Convolutional MobileNet Architecture for Hair Segmentation](https://i.ibb.co/98TMFF9/image.png "Fully Convolutional MobileNet Architecture for Hair Segmentation")

큰 틀에서, FCN_MobileNetV2는 다음과 같이 두 부분으로 나누어 생각할 수 있습니다 (`Contractor` 와 `Decoder` 는 코드에 쓴 이름을 그대로 가져왔습니다):

- `Contractor` 
- `Decoder` 

위 그림에서 초록색 부분까지가 인풋의 H,W 차원을 축소시켜주는 `Contractor` , 노란색부터 아웃풋까지는 축소된 데이터를 다시 원래의 사이즈로 복구시켜주는 `Decoder` 가 됩니다. 여기서 `Contractor` 는 V2 의 몸통을 살짝 변형해서 가져왔고, `Decoder` 는 논문에서 제시한대로 구성하였습니다.



원래 V2라면 마지막 레이어들의 크기가 7 x 7 로 나와야합니다. 하지만 FCN(MobileNetV2)에서는 논문의 구조와 일치시키기 위하여 28 x 28 공간의 레이어들 이후로는 Convolution의 stride를 전부 1로 바꾸어 주었습니다. 그리하여 그림에서 보듯이 중앙의 레이어들이 전부 28 x 28 공간을 유지할 수 있게 됩니다.



## 2. Performance

[Figaro1K](http://projects.i-ctm.eu/it/progetto/figaro-1k) 데이터셋에 50 에폭동안 (K80 GPU 한대 기준으로 약 한시간) 학습시켜본 결과 Train 셋에 대해 평균 90% 이상의 IoU가, Validation 셋에 대해 평균 75% 정도의 IoU를 볼 수 있었습니다. (자세한 사항은 `notebooks/FCN_train.ipynb` 참조)



![alt Example](https://i.ibb.co/wMsFPPW/image.png "Example")
