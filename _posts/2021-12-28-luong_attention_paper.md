---
title: NLP Paper (6) - Luong Attention
categories: [NLP]
comments: true
use_math: true
---



# [Luong Attention] Effective Approaches to Attention-based Neural Machine Translation



**[Luong Attention] 신경망 기계번역에 대한 Attention 기반의 효율적인 접근법**



해당 번역본은 정식 번역본이 아니며 개인이 공부를 위해서 번역한 내용으로 많은 오역, 오타가 존재합니다. 이러한 점을 감안해주시고 읽어주시면 감사하겠습니다.



[alignment](https://en.wikipedia.org/wiki/Bitext_word_alignment) : alignment의 원래 뜻은 '정렬' 이라는 의미이지만 NLP 분야에서는 '단어간의 번역 관계를 식별하는 태스크' 를 의미합니다.

 

## Abstract

- Attention 구조는 신경망 기계번역 분야에서 입력 문장을 번역하는 동안 일부분을 선택적으로 집중하는 방식으로 성능을 향상시키는데 사용되어지고 있습니다.
- 하지만 유용한 구조를 찾아내는데 약간의 작업이 필요합니다.
- 해당 논문이 제안하는 방법은 global 접근법은 항상 모든 입력 단어들을 활용하고, **local한 방법은 오직 입력 단어들의 부분집합을 한번에 활용합니다.**



### Introduction

- Luong 모델은 문장에서 \<eos>라는 단어가 나올 때 까지 입력받고 만약 \<eos>가 나오면 동시에 출력 단어를 만들어냅니다.
- NMT는 커다란 신경망 네트워크로 end-to-end 방식으로 학습되어지고 아주 긴 단어의 시퀀드들도 제대로 일반화시키는 능력을 가졌습니다.
- 최근에 'Attention'이라는 개념이 신경망 네트워크 학습에 자주 사용되며 Bahdanau의 논문에서 이러한 Attention 구조가 성공적으로 적용되었습니다.
- 해당 논문에서는 global 접근법은 모든 입력 단어들을 활용하고 local 접근법은 오직 입력 단어들에 대한 하나의 부분집합을 동시에 고려합니다.
- 전자의 global 접근법은 Bahdanau의 유사한 모델이지만 좀 더 간단한 구조이고, 후자인 **local 접근법은 hard와 soft Attention 모델의 조합**이라고 볼 수 있습니다.
- local 접근법은 global과 soft보다 **계산 비용이 싸고**, **동시에 hard와는 다릅니다.** local 접근법은 대부분의 곳에서 **미분이 가능**하기 때문에 실행과 학습을 만들기 용이합니다.

