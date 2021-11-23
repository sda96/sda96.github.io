---
title: Tensorflow 2 API
categories: [deeplearning]
comments: true
use_math: true

---



## 1. 딥러닝 프레임 워크

딥러닝이라는 기술을 단순히 자체 코드로 구현을 하려면 생각보다 오랜시간이 걸립니다. 특히, 모델의 그래디언트를 수식으로 구하고 이를 바탕으로 역전파를 구현하려면 머리가 아픕니다.

하지만 파이썬에는 딥러닝 기술을 구현시키는데 도움을 주는 다수의 프레임 워크가 존재합니다. 

[2021년 최고의 딥러닝 프레임 워크 Top10](https://em360tech.com/top-10/machine-learning-is-deep)
1. Tensorflow
2. PyTorch
3. Keras
4. Sonnet
5. Caffe
6. Microsoft Cognitive Toolkit
7. MXNet
8. Gluon
9. DL4J
10. ONNX

저희들은 이 중에서 가장 인기가 프레임 워크인 Tensorflow를 사용하겠습니다. 



## 2. Tensorflow

Tensorflow를 간단하게 설명하면 구글에서 배포한 오픈소스 플랫폼으로 모든 종류의 AI개발을 지원해줍니다. 



### 2.1 Tensorflow V1

Tensorflow는 모델의 순전파부분만 설계를 하면 그 모델의 그래디언트(미분값)을 사전에 구해둘 수 있습니다.
이러한 특징은 Tensorflow V1때부터 내려온 ```Directed Acyclic Graph(DAG : 유향 비순환 그래프)``` 덕분입니다.
DAG 덕분에 노드와 노드를 연결하는 매 엣지마다 chain-rule을 기반으로 그래디언트가 역방향으로 전파될 수 있도록 만들어졌습니다.
이러한 방식을 Tensorflow의 ```Graph Mode``` 라고 합니다.

하지만 이러한 설계 때문에 Tensorflow V1은 
- 모델을 구성하는 그래프를 그리는 과정
- 그래프 상에서 연산이 실제 진행되는 과정
2가지 과정을 ```session```을 통하여 엄격하게 분리하였습니다. 그래서 그래프 사이에서 벌어지는 모든 연산은 반드시 ```session.run()```안에서 수행되도록 만들어졌습니다.

이러한 방식이 주는 이점은
- 대규모 분산 환경에서의 확장성
- 대규모 분산 환경에서의 생산성

단점으로는
- 코드가 길고 파이써닉하지 못함
- 구현 방식이 상당히 어려움
- 그래프를 만들고 돌려봐야 비로소 모델 구성 문제가 드러남



### 2.2 Tensorflow V2

하지만 Tensorflow가 V2로 업그레이드되면서 많은 변화가 생겨났습니다.
다양한 변화들중에서 대표적인 것으로
- PyTorch의 ```Eager Mode```를 수용하였습니다.
    - Eager Mode는 딥러닝 그래프가 다 그려지지 않아도 얼마든지 부분실행 및 오류검증이 가능합니다.
    - 코드도 간결하고 파이써닉한 설계를 가졌습니다.
- Keras의 쉽고 간결한 머신러닝 프레임 워크를 수용하였습니다.

이러한 변화들이 있었으며 추가적인 변화 내용은 다음 [링크](https://www.datasciencecentral.com/profiles/blogs/tensorflow-1-x-vs-2-x-summary-of-changes)에서 볼 수 있습니다.


```python
# 텐서플로 1.x
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
# 텐서플로 2.0
outputs = f(input)
```

위의 두 코드는 서로 같은 기능을 하지만 버전에 따른 차이로 Tensorflow V1과 V2의 구현 방식이 크게 간소화된 것을 알 수가 있습니다.

직관적인 코드의 길이 말고도, Tensorflow는 Session.run()에 의존하지 않고, 그래프를 완성하지 않아도 부분 실행이 가능한 Eager Mode의 장점들인 설계, 구현, 디버깅 전과정을 간단하게 만들 수 있도록 바뀐 것을 알 수가 있었습니다.



## 3. TF2 API

딥러닝 기술을 구현할 수 있도록 Tensorflow 에서는 3가지 유형의 API를 제공하고 있습니다.
- Sequential Model
- Functional Model
- Subclassing Model


```python
import tensorflow as tf

# 데이터 로드 및 전처리
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

지금부터는 Tensorflow에서 제공하는 MNIST 데이터셋으로 이미지를 분류하는 다층 퍼셉트론 모델을 동일한 구조로 API만 다르게 구현해보도록 하겠습니다.



### 3.1 Sequential Model


```python
# Sequential Model 설계
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 설정 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습 및 평가
model.fit(x_train, y_train, epochs=1)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"모델의 loss     : {test_loss}")
print(f"모델의 accuracy : {test_acc}")
```

    1875/1875 [==============================] - 4s 2ms/step - loss: 0.4746 - accuracy: 0.8621
    313/313 - 1s - loss: 0.1473 - accuracy: 0.9566
    모델의 loss     : 0.14732250571250916
    모델의 accuracy : 0.95660001039505



### 3.2 Functional API


```python
# Functional 모델 설계
inputs = tf.keras.Input(shape = (28, 28))
x = tf.keras.layers.Flatten()(inputs)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs = inputs, outputs = outputs)

# 모델 설정 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습 및 평가
model.fit(x_train, y_train, epochs=1)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"모델의 loss     : {test_loss}")
print(f"모델의 accuracy : {test_acc}")
```

    1875/1875 [==============================] - 4s 2ms/step - loss: 0.4915 - accuracy: 0.8543
    313/313 - 1s - loss: 0.1515 - accuracy: 0.9537
    모델의 loss     : 0.151547372341156
    모델의 accuracy : 0.9537000060081482



### 3.3 Subclassing API


```python
# Subclassing 모델 설계
class Subclass_model(tf.keras.Model):
    
    def __init__(self):
        super(Subclass_model, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(128, activation = "relu")
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.out = tf.keras.layers.Dense(10, activation = "softmax")
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        return self.out(x)
    
model = Subclass_model()

# 모델 설정 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습 및 평가
model.fit(x_train, y_train, epochs=1)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
print(f"모델의 loss     : {test_loss}")
print(f"모델의 accuracy : {test_acc}")
```

    1875/1875 [==============================] - 4s 2ms/step - loss: 0.4815 - accuracy: 0.8585
    313/313 - 1s - loss: 0.1446 - accuracy: 0.9569
    모델의 loss     : 0.1445624679327011
    모델의 accuracy : 0.9569000005722046



## 4. compile, fit 뜯어보기

모델의 loss와 optimizer를 지정하고 모델의 경과를 지켜볼 수 있는 metrics를 지정해주는 ```model.compile()``` 함수와 입력 데이터들을 배치 단위로 나누어서 모델에 넣어주고 전체 모델의 학습 정도를 조절해주는 ```model.fit()``` 함수를 다른 방식으로 구현해보겠습니다.



### 4.1 model.compile


```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

앞에서 반복적으로 사용한 모델의 학습 방식을 설정하는 ```model.compile()```함수는 매 스텝 학습이 진행될 때 지정된 optimizer로 지정된 loss를 줄이는 과정이 자동으로 진행되어지는데 이 과정을 커스텀마이징이 가능합니다.


```python
# loss function를 SparseCategoricalCrossentropy 로 미리 지정
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
# optimizer를 Adam 으로 미리 지정
optimizer = tf.keras.optimizers.Adam()

# tf.GradientTape()를 활용한 train_step
def train_step(inputs, outputs):
    # 1. 그래디언트 기록 시작
    with tf.GradientTape() as tape:
        # 2. model의 예측값
        predictions = model(inputs)
        # 3. 예측값과 실제값의 loss
        loss = loss_func(outputs, predictions)
        # 4. 학습가능한 가중치별 그래디언트 추출
        gradients = tape.gradient(loss, model.trainable_variables)
    # 5. 추출된 그래디언트 업데이트
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
```

```model.trainable_variables```는 업데이트 해야하는 파라미터들을 지정해주는 역할을 합니다.



### 4.2 model.fit


```python
model.fit(x_train, y_train, epochs=1)
```

실제 학습이 진행되는 함수인 ```model.fit()```는 단순히 데이터를 배치단위로 잘라서 앞에서 만든 학습 함수인```train_step```에 순차적으로 입력하고, 입력하여 나온 loss 와 accuracy 를 반환하는 반복문이라고 생각하면 됩니다.


```python
import time
import numpy as np
from tqdm import tqdm

def train_model(batch_size=32):
    # 시작 시간
    start = time.time()
    # 모델 epoch
    for epoch in range(1):
        x_batch = []
        y_batch = []
        # 모델 step
        for step, (x, y) in tqdm(enumerate(zip(x_train, y_train))):
            # 예제이므로 batch_size번째 데이터만 학습시킴
            if step % batch_size == batch_size -1:
                x_batch.append(x)
                y_batch.append(y)
                loss = train_step(np.array(x_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32))
                x_batch = []
                y_batch = []
        print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))
    print("It took {} seconds".format(time.time() - start))

train_model()
```

    60000it [00:14, 4069.26it/s]
    
    Epoch 0: last batch loss = 0.0009
    It took 14.747461080551147 seconds


​    

