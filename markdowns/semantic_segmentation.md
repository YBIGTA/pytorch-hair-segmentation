# Semantic Segmentation
 **Semantic Segmentation**이란, 이미지를 픽셀별(pixel-wise)로 분류(Classification)하는 것입니다.
 <br>아래의 그림은 Semantic Segmentation의 예시 중 하나입니다.
 
 ![FCN](https://i.ibb.co/YhdKdd5/ss.png)
<br>※ 사진 출처 : https://www.jeremyjordan.me/semantic-segmentation/

오른쪽 사진을 보면, 모든 픽셀이 3가지 클래스(사람, 자전거, 배경) 중 하나로 분류된 것을 확인할 수 있습니다.

## 활용

Semantic Segmentation은 매우 다양한 분야에서 사용되고 있습니다. 대표적인 예로 **자율 주행 자동차**에서 Semantic Segmentation은 핵심적인 역할을 합니다.

![Drive](https://i.ibb.co/TkZYvrD/figure9.png)
<br>※ 사진 출처 : https://devblogs.nvidia.com/image-segmentation-using-digits-5/

자율 주행 자동차가 정면에 위치한 대상이 사람인지, 자동차인지, 횡단보도인지 신호등인지 정확하게 구분하지 못하면 상황에 따른 적절한 판단을 내릴 수 없습니다. 따라서 Semantic Segmentation의 정확도와 속도를 모두 높이기 위해 많은 연구가 이루어지고 있습니다.

## Semantic Segmentation VS Instance Segmentation

![comparison](https://i.ibb.co/0yL6Yjf/is.png)
<br>※ 사진 출처 : http://slazebni.cs.illinois.edu/spring18/lec25_deep_segmentation.pdf

Semantic Segmentation과 Instance Segmentation의 차이를 잘 보여주고 있는 예시입니다. 위의 그림에서 중간에 위치한 **Semantic Segmentation**의 경우, 각 픽셀을 사람(핑크색)과 배경(검은색) 중에 어떤 클래스에 속하는지 분류하고 있습니다. 이와 달리, 오른쪽에 위치한 **Instance Segmentation**은 사람과 배경을 구분해줄 뿐만 아니라 사람끼리도 구분해주고 있는 것을 확인할 수 있습니다.
 즉, Semantic Segmentation은 단순히 각각의 픽셀이 어떤 클래스에 속하는지 분류하는 것에 그치는 반면에 Instance Segmentation은 동일한 클래스에 속하더라도 각각의 사물을 개별적으로 구분해줍니다.
 
## 대표적인 논문

**Semantic Segmentation** 분야의 대표적은 논문들은 아래와 같습니다.

 1. FCN (2014) : https://arxiv.org/abs/1411.4038
 2. U-Net (2015) : https://arxiv.org/abs/1505.04597
 3. SegNet (2015) : https://arxiv.org/abs/1511.00561
 4. PSPNet (2016) : https://arxiv.org/abs/1612.01105
 5. DeepLab V3+ (2018) : https://arxiv.org/abs/1802.02611
