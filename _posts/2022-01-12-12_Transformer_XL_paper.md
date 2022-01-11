---
title: NLP Paper (12) - Transformer-XL
categories: [NLP]
comments: true
use_math: true
---



# [Transformer-XL] Attentive Language Models Beyond a Fixed-Length Context



[**[Transformer-XL] 고정 길이 문맥을 넘어서는 어텐션 기반 언어 모델**](https://arxiv.org/abs/1901.02860)



해당 번역본은 정식 번역본이 아니며 개인이 공부를 위해서 번역한 내용으로 많은 오역, 오타가 존재합니다. 이러한 점을 감안해주시고 읽어주시면 감사하겠습니다.



## Abstract

- 트랜스포머는 고정된 길이의 문장으로 인하여 장기기억의존 문제가 발생하며 이는 언어모델의 성능을 제한시키기 때문에 아래의 2가지 방법을 사용하여 문제를 해결하고자 합니다.
  - segment 수준의 재귀구조를 사용하여 장기기억의존 문제 해결
  - 새로운 포지셔널 인코딩를 사용하여 context fragmentation 문제 해결
- 해당 방법을 적용하면 기존보다 더 빠르게 학습하고 더 긴 내용도 이해하는 모델이 완성됩니다.



### Introduction

- 과거부터 뉴럴넷의 장기기억의존 문제를 해결하기 위해서 RNN의 경우 LSTM이라는 모델이 만들어 졌습니다.

- 지금 현재 트랜스포머도 동일한 문제를 겪고 있으며 트랜스포머의 경우 고정된 길이의 문맥 때문에 장기기억의존 문제에 얽메이게 되었습니다.
- 이 문제를 해결하기 위해서 고정 길이 **segment 단위로 분절하는 방법**을 제안하고 있습니다.
- 하지만 segment 단위로 분절하게 되면 문맥적 정보가 부족하며 context fragmentation 문제가 발생합니다.
- context fragmentation 문제를 해결하기 위해서 **self-attention에 재귀성을 부여**한 재귀 연결성을 추가하였으며 재귀 연결성의 **시간적 일관성을 위한 효율적인 포지셔널 인코딩 구조** 또한 구축하였습니다.

- 장기기억의존성 문제 발생 -> segment 단위로 분절 

  -> context fragmentation 문제 발생 -> self-attention에 재귀성 부여 

  -> 새로운 attention에 맞는 포지셔널 인코딩 구조 구축
