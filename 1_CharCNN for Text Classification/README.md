
# CharCNN for Text Classification
- 논문명: Xiang Zhang, Junbo Zhao, & Yann LeCun. "Character-level Convolutional Networks for Text Classification". (2016).


- 아쉬운점은 pytorch로 구현하지 않고 keras로 구현했다는 점!


## TIL
- OOV_ token
- 오프셋 상수
- 파라미터 계산


## 느낀점
CNN으로도 텍스트 학습 가능하구나~ 역시 멀티 모델러가 되야해!


## 헷갈리는 점 정리

**헷갈렸던 점 1**
- Conv: h(y)에 관해서 헷갈림
논문에서 처음에는 conv 로 h(y)를 계산했는데 f(x)랑 g(y\*d -x+c)를 곱하고 x에 대해 더하는거!
근데 뒤에는 h(y)가 max pooling function으로 나옴.
물론 conv 하는 과정에서 max pooling 과정이 들어가긴 하지만 그래도 수식이 저래도 되는건가?

$$ h(y) = \sum_{x=1}^{k} f(x) \cdot g(y\cdot d-x+c)$$ -(1)
$$h(y) = \max_{x=1}^{k} g(y \cdot d -x +c)$$ -(2)

(1)과 (2)가 같다면, f(x)가 굳이 필요없는 거 아닌가? cnn을 하는데 커널이 필요없어지는건데 사실 말도 안됨.
혹시 1번의 h(y)가 끝나고 다음과정에서 2번의 h(y)가 등장하는건가? 
(Conv 끝내고 풀링하는것처럼, 근데 다른 과정인데 왜 기호를 똑같이 썼지??)
이건 discussion해봐야겠다


**헷갈렸던 점 2**
- 논문 내 table 1. and model.summary() 결과 
  - conv1 input dim: (None,1014,70)
  - conv1 output dim: (None, 1008,256) cf.(batchsize, height,width,channels)에서 우리는 channels 무조건 1
  - 여기서 conv를 했는데 width의 dim이 input보다 커지는 게 이해가 안되었음.
  - 이걸 이해하기 위해서 커널의 dim까지 알고 있어야 하는데, 논문에서 커널 수가 7이라고 해서 나는 당연히 커널 dim이 (1,7)인줄 알았다. 하지만, 사실 커널 dim은 우리의 input dim 70 (알파벳+기호+숫자 수)과 관련이 있었다. 커널 dim은 (7\*70)이었고, (**conv1d라고 해서, 커널사이즈가 1d는 아니다!**)
  - 여기서 256은 커널 개수였다. 즉, 다시 말하면 (7\*70) 커널을 256개를 넣는 다는 뜻. 그래서 width의 dim은 커질 수 밖에 없음
  - 그리고, 커널 내 숫자들은 가중치가 달라지기 때문에 (역전파 학습) feature map의 같은 열 끼리의 숫자는 같지 않다!

 ![image](https://user-images.githubusercontent.com/77769026/148919910-5c0cbdb7-9247-4b1a-8d48-68f47be5d581.png)

