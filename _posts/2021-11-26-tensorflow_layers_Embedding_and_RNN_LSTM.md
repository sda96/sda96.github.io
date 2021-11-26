---
title: Embedding layer와 RNN, LSTM layer
categories: [math]
comments: true
use_math: true

---



## 1.Embedding layer

일상에서 문자와 숫자, 기호등 다양한 문자로 표현되어 지고 있습니다. 이러한 문자형 데이터들을 사용하여 컴퓨터가 문자의 생성, 분류, 예측을 하려면 문자형 데이터들을 컴퓨터가 읽을 수 있는 숫자형 데이터로 변환을 해주어야 합니다.

문자형 데이터를 숫자형 데이터로 바꾸는 노력중 하나가 원-핫 인코딩(One-Hot-Encoding)입니다.
[![image](https://user-images.githubusercontent.com/51338268/143153056-91ca3742-f011-4f31-8ea4-0b793bbf98f6.png)](https://zetawiki.com/wiki/%EC%9B%90-%ED%95%AB_%EC%9D%B8%EC%BD%94%EB%94%A9)
원-핫 인코딩을 통하여 "Red", "Yellow", "Green"과 같은 문자형 데이터를 1과 0으로 이루어진 벡터로 표현을 할 수가 있으며 한 벡터가 가지는 원소의 대부분이 0이기 때문에 희소 표현(Sparse Representation)이라고도 부릅니다.

하지만 희소표현의 문제점은 단어의 개수가 늘어나면 그 만큼 단어들을 표현하기 위한 차원의 수도 늘어나게 되어 공간적 낭비를 불러와 연산량이 늘어나게 됩니다.

[![image](https://user-images.githubusercontent.com/51338268/143159351-d8791d64-3741-49f4-a1eb-50eee436119d.png)](https://www.researchgate.net/figure/Two-different-representations-of-molecular-fragments-a-traditional-sparse-and-one-hot_fig1_323661034)

희소 표현과 반대되는 개념으로 밀집 표현(Dense Representation)이라고 부릅니다.
밀집 표현은 벡터의 차원인 k는 단어의 개수가 아닌 사용자가 설정하는 값이며 원소의 값들은 0과 1이 아닌 실수값으로 표현이 되어집니다.



### Tensorflow Embedding Layer

tensorflow의 embedding layer는 입력으로 들어오는 희소표현을 밀집표현으로 학습시켜주는 layer입니다.


```python
import numpy as np
import tensorflow as tf



inputs = np.random.randint(10, size = (10, 10))
# inputs.shape = (문장의 개수, 문장내의 단어의 개수)
print(f"Embedding Layer 처리 전 output shape : {inputs.shape}")

embedd_layer = tf.keras.layers.Embedding(input_dim = 100, 
                                         output_dim = 2,
                                        )
outputs = embedd_layer(inputs)
print(f"Embedding Layer 처리 후 output shape : {outputs.shape}")
print("\n")
print(f"Embedding Layer 거치기 전 : {inputs[0, :5]}") 
print(f"Embedding Layer 거친 후   : \n{outputs[0, :5].numpy()}")
print("\n")
print(f"Eebedding Layer 학습 파라미터 개수 : {embedd_layer.count_params()}")
```

    Embedding Layer 처리 전 output shape : (10, 10)
    Embedding Layer 처리 후 output shape : (10, 10, 2

    Embedding Layer 거치기 전 : [6 5 1 6 8]
    Embedding Layer 거친 후   : 
    [[ 0.04777438 -0.00765631]
     [-0.03563349 -0.00441667]
     [ 0.04575502 -0.01517576]
     [ 0.04777438 -0.00765631]
     [ 0.026101    0.02042036]]

    Eebedding Layer 학습 파라미터 개수 : 200


[![image](https://user-images.githubusercontent.com/51338268/143161412-073e9cf0-930d-4f77-b246-12d091f72fa3.png)](https://ebbnflow.tistory.com/154)

밀집표현으로 표현된 벡터들은 지속적으로 들어오는 입력값들에 의해서 Embedding에 있는 파라미터들이 학습을 하게 되며 결국에는 지정된 차원으로 이루어진 각 단어들의 밀집 표현을 가진 벡터를 얻을 수 있게 됩니다.



### word2vec

word2vec에 대한 내용의 다수는 cs182 Lecture 13의 내용을 참고하였습니다.

하지만 원-핫 인코딩의 문제점은 문자를 숫자로 표현만 할 수 있을 뿐 문자의 의미나 내용을 내포하고 있지는 않다는 점 입니다.

밀집벡터가 단어의 의미를 내포하기 위해서 **분포 가설**과 **분산 표현**이라는 개념을 사용하였습니다.
- 분포 가설 : 유사한 의미를 가진 단어는 유사한 맥락에서 쓰인다.
- 분산 표현 : 유사한 의미를 가진 두 단어 벡터의 거리를 가깝게 표현된다.

분산 표현을 통하여 비슷한 의미를 가진 단어끼리는 좌표상에서 서로 가까운 위치에 존재하며 분포 가설을 통하여 비슷한 단어끼리는 비슷한 상황에 쓰인다는 단어의 의미와 내용을 포함한 단어 벡터를 얻을 수 있게 되었습니다.

단어의 의미를 포함하는 단어 벡터를 얻기 위한 이론은 이와 같으며 이를 실제로 학습시키는 방법은 **word2vec** 라는 기술이 있습니다.
[![image](https://user-images.githubusercontent.com/51338268/143154419-1be0790f-76fd-4229-8e4a-208416c26928.png)](https://www.researchgate.net/figure/Word2Vec-architecture-The-figure-shows-two-variants-of-word2vec-architecture-CBOW-and_fig1_339921386)
word2vec의 학습 방식은 CBOW, Skip-Gram 구조가 존재합니다.  
word2vec의 모델 구조는 일반적인 딥러닝 모델과 동일하지만 word2vec를 딥러닝 모델이라고 부르지는 않습니다.  
왜냐하면 저희들이 흔히 부르는 딥러닝 모델은 은닉층(Hidden Layer)가 적어도 2개 이상의 층이 쌓여있어야 하기 때문에 word2vec 모델은 shallow learning 모델이라고 부릅니다.



#### word2vec 학습 방법 - CBOW

CBOW는 Continuous Bag of Word의 줄임말로 학습 아이디어는 단어의 의미는 가까이에 있는 주변 단어에 의해서 결정된다는 점에서 시작하였습니다.
[![image](https://user-images.githubusercontent.com/51338268/143155167-cc579f64-67de-4b4b-97e4-2c62a4e92d41.png)](https://cs182sp21.github.io/static/slides/lec-13.pdf)
그림과 같이 가운데에 있는 banking 이라는 단어는 금융, 은행, 제방 쌓기등으로 다양한 뜻을 가졌는데 주변에 있는 단어들을 통하여 가장 자연스럽게 해석이 되는 금융이라는 단어를 선택할 수 있게 됩니다.

이러한 개념에 따라서 CBOW는 주변 단어가 주어졌을 때, 중심 단어를 예측하는 방식으로 학습을 하게됩니다. 



#### word2vec 학습 방법 - Skip - gram

[![image](https://user-images.githubusercontent.com/51338268/143162590-ce5d4d1d-cc1c-4de5-aff9-ec4efa2f62dc.png)](https://cs182sp21.github.io/static/slides/lec-13.pdf)

Skip - gram의 학습 아이디어는 중심 단어를 통하여 주변 단어를 예측하는 방식으로 학습을 진행합니다.  
CBOW보다 예측해야하는 경우의 수나 난이도가 Skip-gram 방식이 더 어렵기 때문에 만약 제대로 학습이 진행되었다면 Skip-grma 방식이 더 성능이 좋은 경우가 많습니다.

![image](https://user-images.githubusercontent.com/51338268/143163441-f901a172-11cb-4fbc-83df-89552248e0af.png)

수식으로 표현하면 중심 단어(c)가 주어졌을 때, 주변 단어(o)가 주어질 조건부 확률이 주변 단어 벡터와 중심 단어 벡터를 내적한 값들의 softmax 를 취한 값과 같습니다.

왜냐하면 softmax 식은 분모가 전체 경우의 수이고 주변 단어중에서 하나의 단어를 의미하고 softmax의 결과는 0~1 사이의 값이기 때문에 확률로써 볼 수 있습니다.

이를 학습하는 구조로 수식을 바꾸면 아래와 같습니다.

![image](https://user-images.githubusercontent.com/51338268/143164577-895f2efd-70d4-469f-9df3-489424877880.png)

중심단어가 주어졌을 때, 나올 수 있는 주변 단어에 대한 확률중에서 가장 높게 나온 주변 단어를 선택하는 방식으로 학습이 진행됩니다.

하지만 해당 수식의 문제점은 모든 주변 단어들을 서치하기 때문에 매 학습마다 많은 양의 연산이 소모된다는 점입니다.

이를 해결하기 위한 아이디어로 중심 단어가 주어졌을 때 주변 단어들이 나올 수 있는 다중 분류 문제에서 이진 분류 문제로 접근을 바꾸고자 합니다.

이진 분류 문제의 접근은 다음과 같습니다.

![image](https://user-images.githubusercontent.com/51338268/143167371-423f2c7b-ca28-4d1b-85a6-59a1d7cfb6e8.png)

이진 분류 문제로 접근을 하면 중심단어 c가 주어졌을 때, 좋은 주변 단어 o가 주어졌느냐? 안주어졌느냐로 바꾸어서 생각이 가능하며 이진 분류 문제로 바꾸었기에 softmax 수식에서 sigmoid 수식으로 바꿀 수 있게 됩니다.

![image](https://user-images.githubusercontent.com/51338268/143167564-43bffb14-b7f4-4064-a7ca-db27a137d2fd.png)

추가적으로 좋은 단어만을 분류하는 모델은 좋은 단어만 나오고 좋지 못한 단어는 아예 나오지 못하면서 나오는 양극화가 생길 수 있고 이로 인한 내적이 커지며 연산이 커질 수 있기 때문에 나쁜 단어를 집어넣는 경우도 추가해주며 이를 negative sampling 이라고 부릅니다.

negative sampling을 하는 경우 사전에 좋지 못한 단어로 label된 단어들 중에서 일부만을 사용하게 됩니다.

최종적으로 Skip-gram 모델이 학습해야하는 수식의 구조는 아래와 같습니다.

![image](https://user-images.githubusercontent.com/51338268/143169535-fb008162-b434-447f-bf18-eb378431f462.png)



### word2vec in Tensorflow

word2vec는 많은 양의 말뭉치 데이터로 학습을 해야하기 때문에 개인마다 따로 학습을 시키기에는 부담이 됩니다.  
이러한 문제를 해결하기 위해서 사전에 이미 학습이 된 모델을 가지고 전이학습을 진행하는 경우가 많은데 사전에 학습된 Embedding vector를 tensorflow의 Embedding layer에 전이시키는 것이 가능합니다.


```python
import gensim

word2vec_path = './model/ko.bin'
word2vec = gensim.models.Word2Vec.load(word2vec_path)
```

사전 학습이 되어있는 word2vec 모델은 해당 [링크](https://github.com/Kyubyong/wordvectors)에서 다운받을 수 있습니다. 만일 불러오는데 에러가 발생하면 ```!pip install --upgrade gensim==3.8.3```로 gensim을 다운그레이드 시켜주시고 커널 재시작을 하면 사용이 가능합니다.


```python
word_vector_dim = 200  # 워드 벡터의 차원수
embedding_matrix = np.random.rand(vocab_size, word_vector_dim)

# embedding_matrix에 Word2Vec 워드 벡터를 단어 하나씩마다 
# 차례차례 카피한다.
for i in range(4,vocab_size):
    if index_word[i] in word2vec:
        embedding_matrix[i] = word2vec[index_word[i]]
embedding_matrix.shape
```

사전에 학습된 Embedding vector를 불러오기 위해서 동일한 크기의 벡터를 만들고 현재 사용하는 단어장안에 존재하는 단어들만 사전학습 모델에 가져와서 새로운 Embedding vector를 만들어 줍니다.  
가져온 Embedding vector의 차원은 200 이므로 저희들이 사용하는 Embedding layer의 차원을 동일하게 맞춰주어야 합니다.

```
tf.keras.layers.Embedding(
            vocab_size,  
            embedding_size, 
            embeddings_initializer=Constant(embedding_matrix),  
            trainable=True,  
            input_length=maxlen  
                         )
```

사전 학습된 Embedding matrix가 완성되면 tensorflow의 Embedding layer의 초기값으로 넣어주고 이를 Fine-tuning을 하고 싶으면 trainable을 True로 하고 기존의 가중치를 그대로 사용하고 싶으면 trainable을 False로 만들어 주면 됩니다.



## 2. SimpleRNN layer

[![image](https://user-images.githubusercontent.com/51338268/143396849-101154f2-1582-4ff6-ae93-eeb488867d2d.png)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)


모델이 입출력값을 내보내는 경우의 수는 그림과 같습니다.

- one-to-one      : 1개의 입력, 1개의 출력의 경우로 일반적인 Dense layer
- one-to-many    : 1개의 입력, 다수의 출력의 경우, Image Cationing
- many-to-one    : 다수의 입력, 1개의 출력, Sentiment Classification
- many-to-many-1 : 다수의 입력, 다수의 출력, Machine Translation
- many-to-many-2 : 다수의 입력, 다수의 출력, Video Classification

지금까지 일반적인 다층 퍼셉트론 모델(dense layer)은 하나의 입력값을 받아서 하나의 출력을 내보내는 one-to-one 방식의 모델이었습니다.

하지만 Sequential한 특성을 가진 데이터(문장, 영상, 음성 데이터)는 기존의 one-to-one 모델로는 '시간의 순서'에 대한 관계성을 학습시키기가 어렵습니다.

왜냐하면 Sequential한 데이터에서 t번째 데이터의 경우 t+1번째 데이터에 큰 영향을 주는 데이터인데 one-to-one 모델은 t번째 데이터와 t+1번째 데이터는 그저 다른 데이터일 뿐, 두 데이터의 관계성을 학습하기가 어렵습니다.  

또한, 입력되는 문장속 단어마다 가지는 의미의 정도가 모두 다른데, 이를 모두 동일한 가중치를 주는 것도 컴퓨터 자원의 낭비입니다.

이러한 관계성을 인지하고 좀 더 효율적인 방법으로 학습시키기 위해서 고안된 구조가 바로 RNN입니다.

RNN은 Recurrent Neural Network의 줄임말로 순환 신경망이라고도 불립니다. 순환 신경망이라고 불리는 이유는 그림과 같습니다.

[![image](https://user-images.githubusercontent.com/51338268/143400011-6bc92edf-1d39-43fb-b2ac-e5b3e27fff03.png)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

그림과 같이 이전의 자신을 호출하는 재귀적인(Recurrent) 형태를 띄고 있기 때문에 순환 신경망이라고 불립니다.

해당 신경망 구조를 좀 더 이해하기 쉽게 펼쳐서 보면 아래 그림과 같습니다.

![image](https://user-images.githubusercontent.com/51338268/143400331-02f95b11-bb55-45a0-abfb-7d96a1317f74.png)

위의 그림에서 A에 해당하는 부분인 하나의 RNN block을 자세하게 살펴보겠습니다.

[![image](https://user-images.githubusercontent.com/51338268/143520066-61f973d5-fc5a-437f-8b21-5e8fa9e1f4eb.png)](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)

위의 그림에서 가중치를 포함한 형태로 다시 그려보겠습니다.

![image](https://user-images.githubusercontent.com/51338268/143522978-0afdc1a2-e635-4d7f-b3bb-862cc62c9b83.png)


$$
\begin{matrix}
H_t &=& tanh(W_{H_tH_{t-1}} + W_{XH}X_t)  \\
Y_t &=& W_{HY}H_t
\end{matrix}
$$

$H_t$는 t시점의 hidden state를 의미하며 $H_{t-1}$는 t시점의 바로 전 시점인 t-1시점의 hidden state를 말합니다.  
hidden state의 수식을 통하여 이전 시점인 t-1의 hidden state를 확인하면서 현재 t 시점의 hidden state에 얼마만큼 영향을 줄지를 학습하는 가중치가 존재합니다.  
학습 가능한 가중치 존재하는 덕분에 시점의 t-1시점이 t시점에 얼만큼의 영향을 주는지 학습할 수 있게 되었습니다.

RNN은 총 3가지의 가중치가 존재하며 가중치의 밑은 왼쪽은 입력으로 들어오는 값을 의미하고 오른쪽은 출력으로 나오는 값을 의미합니다.  
그래서, $W_{xH}$는 x를 입력으로 받고 h로 출력하는 가중치라고 할 수 있습니다.

그리고 t시점의 hidden state에 씌워지는 활성화 함수는 tanh함수입니다.
sigmoid함수가 아니라 tanh함수가 사용되는 이유는 그림을 비교하며 알아보겠습니다.

[![image](https://user-images.githubusercontent.com/51338268/143409838-f69ec315-066c-47b3-9d71-c4c35767fc7a.png)](https://medium.com/@omkar.nallagoni/activation-functions-with-derivative-and-python-code-sigmoid-vs-tanh-vs-relu-44d23915c1f4)

sigmoid 함수의 범위는 0과 1사이이고 미분을 한 범위는 0에서 0.25입니다.

[![image](https://user-images.githubusercontent.com/51338268/143410929-6b3df75c-48b0-4ea2-8b34-4833213ddfe1.png)](https://morioh.com/p/21b55ba475f9)

반면에 tanh 함수의 범위는 -1과 1사이이고 미분을 한 범위는 0에서 1입니다.

tanh 함수는 sigmoid 함수에 비해서 미분한 값의 범위가 넓기 때문에 학습을 비교적 더 길게 유지가 가능합니다.

하지만 tanh를 사용하는 노력에도 불구하고 gradient vanishing 문제는 완벽하게 해결하지 못하였습니다.

RNN에 차례대로 입력값이 들어오는데 가장 먼저 들어온 입력에 대한 정보가 순서에서 멀어질수록 희석되어지게 되면서 결국에는 가중치가 0이 나오게 되면서 더 이상 학습을 하지 못하는 Vanishing Gradient 문제가 발생하게 됩니다.

과거의 데이터에 대한 정보를 잊지 않고 지속적으로 학습을 유지하기 위해서 나온 개선된 구조가 바로 LSTM layer 입니다.



### Simple RNN in Tensorflow


```python
inputs = "What time is it ?"
dic = {
    "is": 0,
    "it": 1,
    "What": 2,
    "time": 3,
    "?": 4
}

print("RNN에 입력할 문장:", inputs)

inputs_tensor = tf.constant([[dic[word] for word in inputs.split()]])

print("Embedding을 위해 단어 매핑:", inputs_tensor.numpy())
print("입력 문장 데이터 형태:", inputs_tensor.shape)

embedding_layer = tf.keras.layers.Embedding(input_dim=len(dic), output_dim=100)
emb_out = embedding_layer(inputs_tensor)

print("\nEmbedding 결과:", emb_out.shape)
print("Embedding Layer의 Weight 형태:", embedding_layer.weights[0].shape)


tf.keras.layers.SimpleRNN(units=64, return_sequences=True, use_bias=False)
rnn_seq_out = rnn_seq_layer(emb_out)

print("\nRNN 결과 (모든 Step Output):", rnn_seq_out.shape)
print("RNN Layer의 Weight 형태:", rnn_seq_layer.weights[0].shape)
```

    RNN에 입력할 문장: What time is it ?
    Embedding을 위해 단어 매핑: [[2 3 0 1 4]]
    입력 문장 데이터 형태: (1, 5)
    
    Embedding 결과: (1, 5, 100)
    Embedding Layer의 Weight 형태: (5, 100)
    
    RNN 결과 (모든 Step Output): (1, 5, 64)
    RNN Layer의 Weight 형태: (100, 64)



## 3. LSTM layer

LSTM layer는 기존의 RNN 구조는 문장의 단어가 순차적으로 들어오면서 처음 순서에 있는 단어의 의미(가중치)가 희석(0에 가까워지 면서)되면서 생기는 vanishin gradient 문제를  Long memory와 Short memory를 나눠서 해결하고자 하여 만들어진 구조입니다.

[![image](https://user-images.githubusercontent.com/51338268/143518770-c46df1bd-3214-4184-9cf5-51d036010a96.png)](https://cs182sp21.github.io/static/slides/lec-10.pdf)


LSTM의 구조는 위의 그림과 같으며 장기 메모리를 가지는 cell state가 새롭게 등장하였고, hidden state는 단기 메모리를 책임지게 됩니다.

cell state가 장기 메모리의 역할을 할 수 있는 이유는 과거 hidden state들에 대한 평균의 의미를 가지기 때문입니다.

평균을 구하는 방식은 일반적으로는 아래의 수식과 같습니다.
$$
c_t = \frac{1}{N} \sum^N_{i = 1}x_i
$$
하지만 평균을 구하는 또 다른 방식은 아래와 같습니다.
$$
\begin{matrix}
c_t &=& \frac{1}{N} \sum^N_{i = 1}x_i \\
&=& \frac{N-1}{N} \frac{1}{N-1}\sum^{N-1}_{i = 1}x_i + \frac{1}{N}x_n \\
&=& \beta c_{t-1} + (1-\beta)x_N, (0 < \beta = \frac{N-1}{N} < 1)
\end{matrix}
$$
또 다른 방식의 평균을 구하는 방식의 형태가 cell state와 닮은 것을 알 수 가 있습니다.
$$
a_t = \beta a_{t-1} + (1-\beta)x_N \\
a_t = f_t * a_{t-1} + i_t * H_t
$$

- $f$는 forget gate
    - $f = sigmoid(W_{hf}h_{t-1} + W_{xf}x_t)$
    - forget gate가 sigmoid를 사용하는 이유는 앞선 $\beta$의 범위가 0에서 1사이이기 때문에 이를 만족하는 활성화 함수가 sigmoid이기 때문입니다.
    - 이전 cell state를 얼만큼 forget(망각) 할 것이냐?
- $i$는 input gate
    - $i = sigmoid(W_{hi}h_{t-1} + W_{xi}x_t)$
    - 새로 만드는 RNN의 hidden state에서는 얼만큼 input(입력) 받을 것이냐?
- $H_t$는 RNN의 hidden state
- $a_{t-1}$는 t-1시점의 cell state

LSTM의 새로운 hidden state는 단기 메모리의 기억을 하는 역할을 가지며 식은 다음과 같습니다.
$$
h_t = o * tanh(a_t)
$$

- $o$는 output gate
    - $o = sigmoid(W_{ho}h_{t-1} + W_{xo}x_t)$
    - 새롭게 만들어진 cell state를 새로운 hidden state에 얼마나 반영할지를 것이냐?
- $a_t$는 cell state 입니다.



### LSTM in Tensorflow


```python
lstm_seq_layer = tf.keras.layers.LSTM(units=64, return_sequences=True)
lstm_seq_out = lstm_seq_layer(emb_out)

print("\nLSTM 결과 (모든 Step Output):", lstm_seq_out.shape)
print("LSTM Layer의 Weight 형태:", lstm_seq_layer.weights[0].shape)

lstm_fin_layer = tf.keras.layers.LSTM(units=64)
lstm_fin_out = lstm_fin_layer(emb_out)

print("\nLSTM 결과 (최종 Step Output):", lstm_fin_out.shape)
print("LSTM Layer의 Weight 형태:", lstm_fin_layer.weights[0].shape)
```


    LSTM 결과 (모든 Step Output): (1, 5, 64)
    LSTM Layer의 Weight 형태: (100, 256)
    
    LSTM 결과 (최종 Step Output): (1, 64)
    LSTM Layer의 Weight 형태: (100, 256)



## 4. 참고사이트

- [희소 표현, 밀집 표현](https://wikidocs.net/33520)
- 그림들의 출처는 그림을 클릭하면 나옵니다.
