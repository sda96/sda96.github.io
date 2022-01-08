---
title: NLP Paper (07) - Transformer
categories: [NLP]
comments: true
use_math: true
---



# [Transformer] Attention Is All You Need



**[Transformer] Attention은 당신이 필요한 모든 것이다!**



해당 번역본은 정식 번역본이 아니며 개인이 공부를 위해서 번역한 내용으로 많은 오역, 오타가 존재합니다. 이러한 점을 감안해주시고 읽어주시면 감사하겠으며 논문의 링크도 남깁니다.

[Transformer paper link](https://arxiv.org/pdf/1706.03762.pdf)

 해당 [링크](https://github.com/sda96/Going_Deeper_Project/blob/main/10_Transformer_translation/10.%20Transformer%EB%A1%9C%20%EB%B2%88%EC%97%AD%EA%B8%B0%20%EB%A7%8C%EB%93%A4%EA%B8%B0.ipynb)는 최대한 논문의 내용만을 따라가며 텐서플로우 프레임 워크를 사용하여 코드로 구현한 내용을 담고 있는 주피터 노트북입니다.



[constituency parsing](https://www.analyticsvidhya.com/blog/2020/07/part-of-speechpos-tagging-dependency-parsing-and-constituency-parsing-in-nlp/) : 문장을 더 작은 서브 구절 단위로 나누어서 분석하는 방법으로 constituents 라고도 불립니다.



## Abstract

- 주요한 시퀀스 번역 모델들은 복잡한 RNN, CNN기반의 인코더 디코더 형태의 신경망 네트워크였습니다.
- 해당 논문에서 새롭고 간단한 구조를 소개하며 그 이름은 'Transformer'입니다. **'Transformer'는 RNN, CNN을 배제한 오직 'Attention'구조를 활용 하였습니다.**
- 해당 방식의 번역 품질은 더 뛰어났고, 더욱 **병렬화가 쉬웠으며**, 학습에 필요한 시간이 훨씬 적었습니다.
- 'Transformer'구조는 영어 서브워드 구문분석에서도 크고, 제한된 훈련 데이터의 양에서도 성공적으로 적용되어져서 다른 태스크에서도 잘 일반화되어집니다.



### 1. Introduction

- 언어모델과 기계번역 분야에서 RNN 기반의 다양한 모델들이 다양한 모습으로 발전해왔습니다.
- RNN 기반의 모델들은 일반적으로 입력 시퀀스와 출력 시퀀스의 단어들 위치에 따라서 계산 비용에 영향을 끼칩니다.
- 이러한 본질적인 시퀀스 환경은 예시들에 따라서 한정된 메모리 제약을 넘어서는 길이의 시퀀스와 같은 훈련 예시들의 병렬화를 배재했습니다.
- 최근까지도 성능의 향상과 계산 효율성이 증가한 모델이 개발되어오고 있지만 시퀀스 연산에 대한 기본적인 제약이 여전히 남아있었습니다.

- 해당 논문에서는 Transformer를 제안하며, 이 모델의 구조는 재귀성을 피하였고, 대신에 **입력과 출력사이의 전체적인 종속성을 만드는데 'Attention' 구조에 전적으로 의존합니다.**



### 2. Background

- 해당 논문에서 시퀀스 연산을 감소키는 것이 다음과 같은 방법을 적용하였습니다.
- **위치에 따른 가중치를 적용한 Attention을 평균화**시키는 방법으로 이 방법은 **Multi-Head Attention** 효과와 대응합니다. 동일합니다.
- **Self-Attention**은 내부 Attention이라고도 부르며 **시퀀스 표현을 계산하기 위한 하나의 시퀀스**로 서로 다른 위치들과 관련된 Attention 구조입니다.



### 3. Model Architecture

![image](https://user-images.githubusercontent.com/51338268/147523979-9fa2a119-ad8a-4322-9ecb-2fa829030843.png)

- Transformer는 기본적으로 Encoder-Decoder 구조를 이루며 과거에는 RNN, LSTM이 적용되던 부분을 Multi-Head Attention으로 바꾼 형태입니다.



#### 3.1 Encoder and Decoder Stacks

**Encoder**

- Encoder layer를 해당 논문에서는 6개를 사용하였으며 Encoder layer를 구성하는 sub-layer는 다음과 같이 구성되어 있습니다.
  - Multi-Head-Attention layer
  - Postion wise fully connected feed forward layer
- 각 sub-layer에는 residual connection과 layer normalization을 적용하였습니다.
  - residual connection을 각 sub-layer에 적용하기 때문에 임베딩의 크기와 동일한 차원으로 유닛수를 유지해주었으며 논문에서는 512로 지정해주었습니다.

**Decoder**

- Decoder layer도 똑같이 6개를 사용했으며 Encoder layer와 유사하지만 사용된 Attention layer의 종류가 2가지입니다.
  - Masked Multi-Head-Attention layer
    - 다음으로 오는 위치값을 보존하기 위해서 일부러 마스킹시키는 변형된 self attention layer 입니다.
  - Multi-Head-Attention layer



#### 3.2 Attention

- Attention 함수는 query와 한 쌍의 key-value를 출력과 맵핑함으로써 입력을 설명해주는 함수입니다.

**Scaled Dot-Product Attention**

![image](https://user-images.githubusercontent.com/51338268/147730815-f130e9ba-f8fb-4611-a991-2c3da618f15f.png)

- Attention 연산에는 곱 연산(Dot-product attention)과 합 연산(Additive attention)이 존재하지만 곱 연산 방식이 더 빠르며 합 연산의 경우에는 layer normalization에서 사용되어 집니다.
- $Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
  - $Q$ : query matrix
  - $K$ : key matrix
  - $V$ : value matrix
  - $d_k$​ : query와 key의 차원의 수

- $\sqrt{d_k}$로 나누어 주었기 때문에 scaled 라고 부르며 scaling을 시킨 이유는 곱 연산으로 너무 커지는 값을 소프트맥스 함수가 작은 미분값을 부여하기 때문에 이를 조정해주기 위해서 처리되었습니다.

**Multi-Head Attention**

![image](https://user-images.githubusercontent.com/51338268/147731475-4f567d8b-0caf-4f7a-8855-6372b4d002a2.png)

- Multi Head Attention은 동일한 차원의 크기를 가진 query와 key의 차원을 h등분 시켜주며 각각 h등분된 query와 key는 선형결합을 시킨 뒤 다시 concat 시키고 다시 선형결합을 시킨 구조입니다.

- 해당 논문에서는 8등분을 시켰으며 이 함수 덕분에 Attention은 병렬화가 가능해졌습니다.

- $Multihead(Q,K,V) = Concat(head_1, \cdots, head_h)W^O$​

  $head_i = Attetnion(QW^Q_i, KW^K_i, VW^V_i)$

**Applications of Attention in ourt Model**

- Decoder부분에 있는 Encoder의 내용과 Decoder의 내용을 받는 Multi-head Attention은 key와 value를 Encoder에서 받으며 query는 Decoder에서 입력 받습니다.
- Encoder의 Multi-head Attention은 기본적으로 self-attention으로 자기 자신한테서 query, key, value를 가져오게 됩니다.
- Decoder의 Masked-Multi-head Attention는 입력값들의 위치 정보에 대한 순서를 학습하기 위해서 마스킹을 적용합니다.



#### 3.3 Position-wise Feed-Forward Networks

- attention sub layer의 경우 활성화 함수가 적용된 적이 없기 때문에 해당 sub-layer에서 활성화 함수인 ReLU를 적용하여 비선형성을 부여해줍니다.
- $FFN(x) = \max(0, xW_1 +b_1)W_2 + b_2$​



#### 3.4 Embedding and Softmax

- 사전학습된 임베딩 벡터를 사용하며 모델이 vocab_size의 차원중에서 가장 높은 확률을 가진 단어를 도출하기 위한 softmax를 출력층에 적용합니다.



#### 3.5 Postional Encoding

- RNN, CNN모델을 배재하면서 시퀀스 데이터의 연속성, 재귀성을 사용하지 못하게 되면서 입력 데이터의 순서를 알기 어려워졌습니다.

- 입력 데이터의 순서를 부여하기 위한 방법으로 Postional Encoding 입력 데이터에 더해주는 방법을 제안합니다.

- $PE_{(pos, 2i)} = \sin(pos/1000^{2i/d_model})$

  $PE_{(pos, 2i+1)} = \cos(pos/1000^{2i/d_model})$​



### 4. Why Self-Attention

- CNN, RNN layer를 배재하고 Self Attention을 활용하는 이유는 다음과 같습니다.

  - 각 층마다의 계산 복잡도가 비교적 감소하였습니다.

  - 막대한 양의 계산을 병렬화시킬 수 있습니다.

  - 네트워크에서 문장의 길이에 대한 종속성을 통과시켜줍니다.

  - 추가적으로 Self Attention은 해석에 용이한 모델을 제공해줄 수 있습니다.

    ![image](https://user-images.githubusercontent.com/51338268/147733042-2abcad9c-6919-4370-9b42-0b7027d66869.png)



### 5. Training

- Transformer 모델의 성능을 검증하기 위해서 standard WMT 2014 EN-GE 데이터셋과 EN-FR 데이터셋을 사용하여 성능을 비교해보았습니다.
  - sentence encoding : byte-pair encoding, word-piece
  - hardware : 8개의 NVIDA P100 GPU, 12시간, 100,000 steps
  - optimizer : Adam
  - lr $ = d^{-0.5}_{model}\cdot\min(step\_num^{-0.5}, step\_num\cdot warmup\_steps^{-1.5})$​​
  - regularization
    - dropout : 모든 sub-layer가 연산하기 전에 적용한 rate는 0.1
    - [label smoothing](https://3months.tistory.com/465)
    - [beam search (Machine Translation)](https://blog.naver.com/PostView.nhn?blogId=sooftware&logNo=221809101199&from=search&redirect=Log&widgetTypeCall=true&directAccess=false)

