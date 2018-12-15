<p align="center"> <img src="https://i.ibb.co/NmBHNcP/deeplab-v3.png" width="800" height="260"></p>
Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, DeepLab V3+ (2018.08.22)

**DeepLab**은 v1부터 가장 최신 버전인 v3+까지 총 4개의 버전이 있습니다.

> 1\. [<u>DeepLabv1</u>](https://arxiv.org/abs/1412.7062) (2015) : Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs 
> <br>2\. [<u>DeepLabv2</u>](https://arxiv.org/abs/1606.00915) (2017) : DeepLab : Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
> <br>3\. [<u>DeepLabv3</u>](https://arxiv.org/abs/1706.05587) (2017) : Rethinking Atrous Convolution for Semantic Image Segmentation
> <br>4\. [<u>DeepLabv3+</u>](https://arxiv.org/abs/1802.02611) (2018) : <span style="font-size: 10pt;">Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation</span>

이 중 가장 최신의 논문이며 뛰어난 성능을 보이고 있는 DeepLabv3+에 대해서 다루도록 하겠습니다.

<br>


# Abstract

<span style="font-size:10pt;">1\.</span> **<span style="font-size: 10pt;">Spatial pyramid pooling module</span>**<span style="font-size: 10pt;">과</span> **<span style="font-size: 10pt;">Encode-decoder structure</span>** <span style="font-size: 10pt;">각각의 장점을 합쳐 더욱 더 좋은 성능을 보이는 DeepLabv3+를 제시합니다.</span>
<br>- Spatial pyramid pooling module : Rate가 다른 여러 개의 atrous convolution을 통해 다양한 크기(multi-scale)의</span><span style="font-size: 10pt;"> 물체 정보를 encode할 수 있습니다.</span>
<br>- Encode-decoder structure : 공간 정보(spatial information)를 점진적으로 회복함으로써 더욱 정확하게 물체의</span> <span style="font-size: 10pt;">바운더리를 잡아낼 수 있습니다.</span>

<span style="font-size:10pt;">2\.</span> **<span style="font-size: 10pt;">Xception model</span>**<span style="font-size: 10pt;">을 backbone network로 사용합니다.</span>

<span style="font-size: 10pt;">3</span><span style="font-size:10pt;">.</span> **<span style="font-size: 10pt;">The depthwise separable convolution</span>**<span style="font-size: 10pt;">을 통해 encoder-decoder를 더욱 빠르고 정확하게 합니다.</span>

<span style="font-size: 10pt;">4\. 후처리(post-processing) 작업 없이 PASCAL VOC 2012와 Cityscapes의 test data에 대해 각각 89.0%와 82.1%의 성능을 보입니다.</span>
<br>- v2까지는 모델의 output에 CRF를 거치는 post-processing을 통해 더욱 정확한 바운더리를 잡아내었습니다.</span>

<br>
<br>

## 1\. Introduction</span>

- Atrous convolution을 통해 extracted encoder features의 resolution을 임의로 할 수 있습니다. 이는 기존의 Encoder-decoder model에서는 불가능했습니다.
나머지 내용은 모두 Abstract에서 언급한 것과 동일하기 때문에 생략하도록 하겠습니다.

<br>
<br>

## 2\. Related Work

### Atrous Convolution
<p align="center"> <img src="https://i.ibb.co/LtLCFVq/atrous.png" width="400" height="200">
<br>※ 사진 출처 : https://www.mdpi.com/2072-4292/9/5/498/htm)
</p>
Atrous에서 trous는 구멍(hole)을 의미합니다. 즉, convolution에서 중간중간을 비워두고 특정 간격의 픽셀에 대해서 합성곱을 한다고 생각할 수 있습니다. Atrous convolution에는 rate라는 parameter가 존재하는데 사진에서 r이 rate에 해당합니다. r이 커질수록 필터는 더 넓은 영역을 담을 수 있게 됩니다.

<br>장점 : 동일한 수의 파라미터를 이용함에도 불구하고 필터의 receptive field를 키울 수 있습니다. 따라서 semantic segmentation에서는 디테일한 정보가 중요하므로 여러 convolution과 pooling 과정에서 디테일한 정보가 줄어들고 특성이 점점 추상화되는 것을 어느정도 방지할 수 있습니다.


<p align="center"> <img src="https://i.ibb.co/ykwgQfL/fig1.png" width="700" height="500"></p>

> (a) Spatial Pyramid Pooling
> <br>- 다른 rates의 atrous convolution을 parallel하게 적용하여 다양한 크기(multi-scale)의 정보를 담을 수 있습니다.
> <br>
> <br>(b) Encoder-Decoder
> <br>- Encoder-decoder는 human pose estimation, object detection, 그리고 semantic segmentation을 포함해 다양한 컴퓨터 비전 영역에서 좋은 성능을 보여왔습니다. Encoder 부분은 점점 feature maps을 줄여 higher semantic information을 잡아낼 수 있습니다. 또한 Decoder 부분은 점진적으로 공간 정보(spatial information)를 회복합니다.

이 논문에서는 encoder module로 이전 버전인 DeepLabv3을 사용할 것을 제시하고 있습니다.

<br>
<br>
</br>
### Depthwise separable convolution

Depthwise separable convolution은 <u>[Mobilenet](https://arxiv.org/abs/1704.04861)</u> 논문에 담긴 사진과 <u>[링크](http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/)</u>에서 사용한 사진을 통해 정리하도록 하겠습니다.

<p align="center"> <img src="https://i.ibb.co/6NfYrKx/normalconv.png" width="400" height="300"></p>
위의 그림은 일반적으로 사용되는 convolution을 나타낸 사진입니다.

하나의 3x3 filter가 input channel과 동일한 channel을 가지고 합성곱을 하면 channel이 1인 output이 나오게 됩니다.

이러한 filter를 원하는 output channel의 개수만큼 사용합니다.


<p align="center"> <img src="https://i.ibb.co/s9LHRGK/stdfilter.png" width="500" height="200"></p>
이를 일반화하여 나타내면 위와 같습니다.  

Dk는 filter의 크기이고 M은 input channel, 그리고 N은 output channel (= filter의 개수)를 의미합니다.

<br><br>

<p align="center"> <img src="https://i.ibb.co/V3xZ0BV/depthconv.png" width="850" height="300"></p>

Depth separable convolution은 두 가지 과정을 거칩니다.
>1\. Depth Convolution : filter의 channel이 input channel과 같은 것이 아니라 1로 두어서 convolution 결과 channel의 변화가 생기지 않도록 한다.
><br>
>2\. Pointwise Convolution : 1x1 filter를 원하는 output channel의 개수만큼 사용하여 channel의 개수를 맞춰준다.

<p align="center"> <img src="https://i.ibb.co/7k5mHj3/depthfilter.png" width="450" height="200"></p>
<p align="center"> <img src="https://i.ibb.co/B6RhWpc/pointwisefilter.png" width="500" height="200"></p>


Depthwise convolution 과정에서 사용되는 filter는 channel이 1이고 filter의 개수가 input channel인 M과 같습니다.
그리고 Pointwise convolution에서 사용되는 filter는 channel이 input channel인 M이고
filter의 개수가 원하는 output channel과 같습니다.  

Depth Separable Convolution의 장점으로는
일반적인 Convolution과 비슷한 과정을 거치지만 사용되는 parameter의 수를 획기적으로 줄여
모델의 속도를 향상시킬 수 있습니다.

※ 잘 이해가 되지 않으시는 분은 <u>[PR-044: MobileNet](https://www.youtube.com/watch?v=7UoOFKcyIvM&feature=youtu.be)</u>을 참고하시면 좋을 것 같습니다!

## 3. Methods

<p align="center"> <img src="https://i.ibb.co/p4mFdBy/fig2.png" width="700" height="450"></p>

### 3.1 Encoder-Decoder with Atrous Convolution

DeepLabv3+에서는 Encoder로 DeepLabv3을 사용하고 Decoder로 bilinear upsampling 대신에 U-net과 유사하게 concat해주는 방법을 사용합니다.

1) **Encoder (DeepLabv3)** : DCNN(Deep Convolutional Neural Network)에서 Atrous convolution을 통해 임의의 resolution으로 특징을 뽑아낼 수 있도록 합니다. 여기서 output stride의 개념이 쓰이는데 'input image의 resolution과 최종 output의 resolution의 비'로 생각하시면 됩니다. (The ratio of input image spatial resolution to the final output resolution) 즉, 최종 feature maps이 input image에 비해 32배 줄어들었다면 output stride는 32가 됩니다. Semantic segmentation에서는 더욱 디테일한 정보를 얻어내기 위해서 마지막 부분의 block을 1개 혹은 2개를 삭제하고 atrous convolution을 해줌으로써 output stride를 16혹은 8로 줄입니다. 

 그리고 다양한 크기의 물체 정보를 잡아내기 위해 다양한 rates의 atrous convolution을 사용하는 ASPP(Atrous Spatial Pyramid Pooling)를 사용합니다.
 
 <br>2) **Decoder** : 이전의 DeepLabv3에서는 decoder 부분을 단순히 bilinear upsampling해주었으나 v3+에서는 encoder의 최종 output에 1x1 convolution을 하여 channel을 줄이고 bilinear upsampling을 해준 후 concat하는 과정이 추가되었습니다. 이를 통해 decoder 과정에서 효과적으로 object segmentation details을 유지할 수 있게 되었습니다.


<br>
### 3.2 Modified Aligned Xception

<p align="center"> <img src="https://i.ibb.co/1rr4qqZ/xception.png" width="600" height="600"></p>
 DeepLabv3+에서는 Xception을 backbone으로 사용하지만 MSRA의 Aligned Xception과 다른 3가지 변화를 주었습니다.
 
>1) 빠른 연산과 메모리의 효율을 위해 entry flow network structure를 수정하지 않았습니다.
>2) Atrous separable convolution을 적용하기 위해 모든 pooling operation을 depthwise separable convolution으로 대체했습니다.
>3) 각각의 3 x 3 depthwise convolution 이후에 추가적으로 bath normalization과 ReLU 활성화 함수를 추가해주었습니다.


## 4\. Experimental Evalution</span>

<p align="center"> <img src="https://i.ibb.co/cJtMkfR/error.png" width="470" height="160"></p>

DeepLabv3에서 사용했던 ResNet대신에 Xception을 backbone으로 사용하였을 때
Error rate가 더 낮음을 확인할 수 있습니다.

<p align="center"> <img src="https://i.ibb.co/HN60qBw/decoder-effect2.png" width="490" height="350"></p>
<p align="center"> <img src="https://i.ibb.co/gyg68Lj/decoder-effect.png" width="490" height="290"></p>


표를 통해 Decoder를 추가한 것이 훨씬 높은 mIOU를 보인다는 것을 확인할 수 있습니다.

또한 예시를 통해 살펴보면 중간 사진은 DeepLabv3과 동일하게 단순히 BU(Bilinear Upsampling)만 사용한 것이고 오른쪽 사진은 U-net과 유사하게 concat 과정이 있는 decoder를 사용한 것입니다.
이를 통해 Decoder 과정에서 boundary information을 잘 잡아내었음을 확인할 수 있습니다.

## 5\. Conclusion

 DeepLab은 segmentation task의 문제점에 대한 해결책을 찾으며 v1부터 꾸준하게 발전해온 모델입니다. 또한 Atrous convolution과 decoder를 통해 segmentation을 더욱 더 정확하게 할 수 있음을 보여주었습니다.  

## 6\. Reference

<span style="font-size: 14pt;"><span style="font-size: 10pt;"> </span><span style="font-size: 10pt;">1. </span></span><span style="font-size:11pt;"><span style="font-size: 10pt;">DeepLabv3+ 리뷰 : </span>[<span style="font-size: 10pt;">https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/</span>](https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/)</span>

<span style="font-size: 10pt;"> 2\. Atrous Convolution : </span><span style="font-size: 14.6667px;">[<span style="font-size: 10pt;">https://www.mdpi.com/2072-4292/9/5/498/html</span>](https://www.mdpi.com/2072-4292/9/5/498/html)</span>

<span style="font-size: 10pt;"> 3\. MobileNet : </span>[<span style="font-size: 10pt;">https://arxiv.org/abs/1704.04861</span>](https://arxiv.org/abs/1704.04861)<span style="font-size: 10pt;"> (Paper)</span>
<br><span style="font-size: 10pt;"></span> [<span style="font-size: 10pt;">http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/</span>](http://machinethink.net/blog/googles-mobile-net-architecture-on-iphone/)
<span style="font-size: 10pt;">
</span>[<span style="font-size: 10pt;">https://www.youtube.com/watch?v=7UoOFKcyIvM&feature=youtu.be</span>](https://www.youtube.com/watch?v=7UoOFKcyIvM&feature=youtu.be) <span style="font-size: 10pt;">(PR-044)</span>