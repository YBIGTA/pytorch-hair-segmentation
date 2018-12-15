## Visdom

[Visdom](https://github.com/facebookresearch/visdom)은 데이터를 시각화하기 위한 툴입니다. torch, numpy를 지원하고 있습니다.

### 쉬운 시작

1. Visdom server 켜기

```
pip install visdom
python -m visdom.server
# localhost:8097에 접속합니다.
```

2. Visdom에 로그 남기기
```python
DEFAULT_PORT = 8097
DEFAULT_HOSTNAME = "http://localhost"

vis = visdom.Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

vis.text('텍스트 써보기')
vis.images(img:numpy_type) # numpy type의 이미지를 변수에 할당
vis.matplot(plt) # matplotlib의 plot type의 변수에 할당
```

자세한 사항은 [Visdom 문서](https://github.com/facebookresearch/visdom)를 참조해주세요