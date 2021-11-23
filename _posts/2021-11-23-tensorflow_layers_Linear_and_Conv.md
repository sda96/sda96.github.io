---
title: 텐서플로우의 선형층과 합성곱층
categories: [deeplearning]
comments: true
use_math: true
---

## Linear layer

Linear layer, 즉 선형 결합층 불리는 이유는 선형대수학에서 쓰이는 선형 변환이라는 개념과 동일한 기능을 하기에 붙혀진 이름입니다. Linear layer는 이외에도 다양한 이름으로 불리우고 있습니다.
- Fully Connected Layer
- Feadfoward Neural Network
- Multilayer Perceptrons
- Dense Layer

다양한 이름으로 불리고 있지만 결국에는 입력값과 가중치를 내적한 후 편향을 더한 수식을 말하는 것 입니다. 

$$y = Wx + b$$

해당 수식에서 가중치인 W와 편향인 b는 학습 가능한 파라미터로 해당 값들에 변화를 줌으로써 loss가 최소화되는 방향인 예측값 y는 실제값에 가까워지도록 만들어 줍니다.




### Tensorflow의 Linear layer

tensorflow에서 Linear layer의 역할을 하는 layer는 Dense layer 입니다.


```python
import tensorflow as tf


inputs = tf.zeros((64, 10, 20))
# (데이터 개수, 높이, 넓이)
print(f"입력값 shape : {inputs.shape}")


dense_layer = tf.keras.layers.Dense(30)
outputs = dense_layer(inputs)
dense_weight, dense_bias = dense_layer.weights

print(f"가중치 shape : {dense_weight.shape}")
print(f"편향   shape : {dense_bias.shape}")
print(f"출력값 shape : {outputs.shape}")
```

    입력값 shape : (64, 10, 20)
    가중치 shape : (20, 30)
    편향   shape : (30,)
    출력값 shape : (64, 10, 30)


Dense layer의 학습 파라미터의 개수는 각 차원의 개수를 모두 곱한 값에 편향의 차원을 더한 수치로 현재 Dense layer를 기준으로는 다음과 같습니다.

$(20 \times 30) + 30 = 630 $


```python
tf.keras.layers.Layer.count_params(dense_layer)
```


    630

해당 layer의 학습 파라미터의 수를 세는 함수를 사용하여도 630개라는 값이 나오는 것을 알 수가 있습니다.

지금까지는 Dense layer를 사용하는 이유와 내부의 파라미터들의 형태가 어떻게 생겼는지 확인하였습니다. 하지만 tensorflow의 Dense layer의 내부에는 이외에도 다양한 딥러닝 기술들이 구현되어 있습니다.

```
tf.keras.layers.Dense(
    units, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', 
    kernel_regularizer=None,
    bias_regularizer=None, 
    activity_regularizer=None, 
    kernel_constraint=None,
    bias_constraint=None, 
    **kwargs
)
```

위의 함수는 tensorflow 공식문서에서 Dense layer가 가지고 있는 arguments들과 그에 따른 지정된 default값들 입니다.



#### Dense layer arguments 

- units : 출력되는 차원의 크기를 말합니다.
- activations : 지정된 활성화 함수 사용하며 None이면 활성화 함수를 사용하지 않습니다.
- use_bias : 불린값이 들어오며 layer에서 편향을 사용할지 결정합니다.
- kernel_initializer : 커널 가중치 행렬의 초기값을 지정합니다.
- bias_initializer : 편향의 초기값을 지정합니다.
- kernel_regularizer : 커널 가중치 행렬에 regularization 함수를 적용시킵니다.
- bias_regularizer : 편향 벡터에 regularization 함수를 적용시킵니다.
- activity_regularizer : 출력값에 regularization 함수를 적용시킵니다.
- kernel_constraint : 커널 가중치 행렬에 constraint(제한 조건) 함수를 적용시킵니다.
- bias_constrain : 편향 벡터에 constraint 함수를 적용시킵니다.



## Convolutional layer

dense layer는 이미지 분야에서도 사용이 되었지만 컬러 이미지의 경우 color channel 때문에 RGB로 기존 파라미터 개수에 3배로 늘어나게 됩니다.
늘어난 파라미터의 개수는 학습시켜야 하는 계산량이 급격히 늘어나게 되고 시간또한 오래 걸리게 됩니다.

[![image](https://user-images.githubusercontent.com/51338268/142982600-67f86aaa-48fe-4274-83a2-4ca7172384dd.png)](https://wikidocs.net/64066)


이때, Convolutional layer, 합성곱 층이라고도 불리는 conv layer는 사람의 시각 신경이 이미지를 보듯이 하나의 이미지를 **window**, **filter** 이라고 부르는 **kernel** 로 입력 이미지를 순차적으로 sliding 하면서 합성곱 연산을 진행하여 기존 이미지의 특성들을 뽑아냅니다.

이때, **kernel** 이 이동하는 칸 수를 **stride** 라고 부르고 합성곱 연산 결과 나온 출력 결과를 **특성맵(feature map)**이라고 부릅니다.

해당 과정을 시각적으로 잘 표현한 [사이트](https://stanford.edu/~shervine/l/ko/teaching/cs-230/cheatsheet-convolutional-neural-networks)가 있으니 참고하시면 이해가 쉽게 되실거라고 생각됩니다.

이미지를 지나온 filter는 이미지에서 집중하고 있는 부분이 있습니다. 이러한 filter들을 여러개를 사용하면 이미지의 다양한 부분의 특성을 집약시킬 수 있게 됩니다.



### Tensorflow의 Conv layer


```python
import tensorflow as tf

batch_size = 64
pic = tf.zeros((batch_size, 300, 300, 3))
print("Convolution 결과 전:", pic.shape)

conv_layer = tf.keras.layers.Conv2D(filters = 16,
                                    kernel_size = (5, 5),
                                    strides = (1, 1)
                                   )
conv_out = conv_layer(pic)

print("Convolution 결과 후:", conv_out.shape)
print("Convolution Layer의 Parameter 수:", conv_layer.count_params())

pad_conv_layer = tf.keras.layers.Conv2D(filters = 16,
                                    kernel_size = (5, 5),
                                    strides = (1, 1),
                                    padding = "same"
                                   )
pad_conv_out = pad_conv_layer(pic)

print("padding Convolution 결과 후:", pad_conv_out.shape)
print("padding Convolution Layer의 Parameter 수:", pad_conv_layer.count_params())
```

    Convolution 결과 전: (64, 300, 300, 3)
    Convolution 결과 후: (64, 296, 296, 16)
    Convolution Layer의 Parameter 수: 1216
    padding Convolution 결과 후: (64, 300, 300, 16)
    padding Convolution Layer의 Parameter 수: 1216


conv layer에서 padding 이라는 arguments가 존재하는데 **padding**이란 이미지의 경계면에 0을 추가하여 짤릴 수도 있는 이미지의 경계면을 학습시키거나 안정적인 출력값의 shape를 유지시키기 위해서 사용되어집니다.

padding을 추가하면서 파라미터가 증가할거라고 생각이 되지만 결과적으로는 파라미터의 수는 동일하다는 것을 알 수가 있습니다.

그 이유는 conv layer의 파라미터를 구하는 수식은 $(F \times F \times C + 1) \times K$로 padding은 파라미터의 개수에 영향을 주지 않기 때문입니다.
- $F$는 filter 의 크기
- $C$는 입력되는 채널(filter, kernel)의 개수
- $K$는 출력되는 채널의 개수


```python
(5 * 5 * 3 + 1) * 16
```


    1216

그리고 패딩의 여부에 따라서 출력값의 크기가 다른 이유는 수식이 다음과 같이 때문입니다.

$$
O = \frac{I - F + 2P}{S} + 1
$$
- $O$은 출력값의 크기
- $I$는 입력값의 크기
- $F$는 filter의 크기
- $P$는 padding의 크기
- $S$는 스트라이드의 크기


```python
(300 - 5 + 4)/ 1 + 1
```


    300.0

tensorflow 공식문서에서 conv2D 함수가 가지는 args를 살펴보겠습니다.  
일부 겹치는 args의 기능은 dense layer와 겹치므로 중요한 args만 살펴보겠습니다.

```
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```



#### Conv2D layer arguments

- filters : 출력되는 차원의 크기로 사용될 filter의 개수를 의미합니다.
- kernel_size : filter의 크기를 정합니다
- strides : filter가 이미지를 이동하는 칸 수를 지정합니다.
- padding : "valid"와 "same"이 존재하며 valid는 패딩을 하지 않는 것을 말하고 same은 입력값의 상하좌우의 끝에 0을 추가해줍니다. 
    - padding을 same으로 하고 stride를 1로 하면 입력과 동일한 크기의 특성맵(feature map)이 나오게 됩니다.



## Pooling layer

Pooling layer는 conv layer와 함께 자주 사용되는 layer로 흔히, conv layer와 pooling layer가 함께 쓰인 모델을 CNN 모델이라고 부릅니다.

이미지의 정보를 집약시킨 특성맵에서 더욱 핵심적인 정보만 뽑아내는 역할을 합니다.

핵심적인 정보를 뽑는 기준으로는 가장 큰값 MAX로 뽑을 수도 있고 전체의 평균 Avg으로 뽑을 수도 있습니다.

[![image](https://user-images.githubusercontent.com/51338268/142987463-6ab4de65-a42f-4aea-8db9-e9aa108d3c88.png)](https://hobinjeong.medium.com/cnn%EC%97%90%EC%84%9C-pooling%EC%9D%B4%EB%9E%80-c4e01aa83c83)

MAX pooling의 경우 특성맵에서 2x2 크기의 새로운 filter로 stride가 2인 새로운 특성맵을 만든다고 할 때, 기존의 특성맵에서 가장 큰 값만 뽑아서 사용합니다.



### Tensorflow의 Pooling layer


```python
batch_size = 64
pic = tf.zeros((batch_size, 300, 300, 3))
print("Convolution 결과 전:", pic.shape)

conv_layer = tf.keras.layers.Conv2D(filters = 16,
                                    kernel_size = (5, 5),
                                    strides = (1, 1)
                                   )
conv_out = conv_layer(pic)

print("Convolution 결과 후:", conv_out.shape)
print("Convolution Layer의 Parameter 수:", conv_layer.count_params())

pool_layer = tf.keras.layers.MaxPool2D()
pool_out = pool_layer(conv_out) 

print("Pooling 결과 후:", pool_out.shape)
print("Pooling Layer의 Parameter 수:", pool_layer.count_params())
```

    Convolution 결과 전: (64, 300, 300, 3)
    Convolution 결과 후: (64, 296, 296, 16)
    Convolution Layer의 Parameter 수: 1216
    Pooling 결과 후: (64, 148, 148, 16)
    Pooling Layer의 Parameter 수: 0


Pooling을 하고나서의 결과는 shape는 절반으로 감소하였으며 특징적인 점은 Pooling layer는 학습하는 파라미터가 존재하지 않습니다.

이외에도 Pooling이 가지는 효과들에 대해서 알아보겠습니다.



#### translational invariance 효과

이미지는 약간의 상하좌우의 변화가 생긴다고 해도 동일한 이미지인데 컴퓨터는 다르다고 인지합니다.  

Max Pooling을 통해 인접한 영역 중 가장 특징이 두드러진 영역 하나를 뽑는 것은 오히려 약간의 시프트 효과에도 불구하고 동일한 특징을 잡아내는데 도움을 줍니다.

이는 오히려 object 위치에 대한 오버피팅을 방지하고 안정적인 특징 추출 효과를 가져온다고 합니다.



#### Non-linear 함수와 동일한 피처 추출 효과

Relu와 같은 Non-linear 함수도 마찬가지로 많은 하위 레이어의 연산 결과를 무시하는 효과를 발생시키지만, 그 결과 중요한 피처만을 상위 레이어로 추출해서 올려줌으로써 결과적으로 분류기의 성능을 증진시키는 효과를 가집니다. 

Min/Max Pooling도 이와 동일한 효과를 가지게 됩니다.



#### Receptive Field 극대화 효과

Max Pooling이 없이도 Receptive Field를 크게 하려면 Convolutional 레이어를 아주 많이 쌓아야 합니다.  
그 결과 큰 파라미터 사이즈로 인한 오버피팅, 연산량 증가, Gradient Vanishing 등의 문제를 감수해야 합니다.  
이런 문제를 효과적으로 해결하는 방법으로 꼽히는 두 가지 중 하나가 Max Pooling 레이어 사용입니다. 다른 하나로는 Dilated Convolution이 있습니다. 상세한 내용은 다음 [링크](https://m.blog.naver.com/sogangori/220952339643)를 참고하세요.
