---
title: NLP Paper (7) - Transformer
categories: [NLP]
comments: true
use_math: true
---



# [Transformer] Attention Is All You Need



**[Transformer] Attention은 당신이 필요한 모든 것이다!**



해당 번역본은 정식 번역본이 아니며 개인이 공부를 위해서 번역한 내용으로 많은 오역, 오타가 존재합니다. 이러한 점을 감안해주시고 읽어주시면 감사하겠습니다.

 

[constituency parsing](https://www.analyticsvidhya.com/blog/2020/07/part-of-speechpos-tagging-dependency-parsing-and-constituency-parsing-in-nlp/) : 문장을 더 작은 서브 구절 단위로 나누어서 분석하는 방법으로 constituents 라고도 불립니다.



## Abstract

- 주요한 시퀀스 번역 모델들은 복잡한 RNN, CNN기반의 인코더 디코더 형태의 신경망 네트워크였습니다.
- 해당 논문에서 새롭고 간단한 구조를 소개하며 그 이름은 'Transformer'입니다. **'Transformer'는 RNN, CNN을 배제한 오직 'Attention'구조를 활용 하였습니다.**
- 해당 방식의 번역 품질은 더 뛰어났고, 더욱 병렬화가 쉬웠으며, 학습에 필요한 시간이 훨씬 적었습니다.
- 'Transformer'구조는 영어 서브워드 구문분석에서도 크고, 제한된 훈련 데이터의 양에서도 성공적으로 적용되어져서 다른 태스크에서도 잘 일반화되어집니다.



### Introduction

- 언어모델과 기계번역 분야에서 RNN 기반의 다양한 모델들이 다양한 모습으로 발전해왔습니다.
- RNN 기반의 모델들은 일반적으로 입력 시퀀스와 출력 시퀀스의 단어들 위치에 따라서 계산 비용에 영향을 끼칩니다.
- 이러한 본질적인 시퀀스 환경은 예시들에 따라서 한정된 메모리 제약을 넘어서는 길이의 시퀀스와 같은 훈련 예시들의 병렬화를 배재했습니다.
- 최근까지도 성능의 향상과 계산 효율성이 증가한 모델이 개발되어오고 있지만 시퀀스 연산에 대한 기본적인 제약이 여전히 남아있었습니다.

- 해당 논문에서는 Transformer를 제안하며, 이 모델의 구조는 재귀성을 피하였고, 대신에 **입력과 출력사이의 전체적인 종속성을 만드는데 'Attention' 구조에 전적으로 의존합니다.**



### Background

- 해당 논문에서 시퀀스 연산을 감소키는 것이 다음과 같은 방법을 적용하였습니다.
- **위치에 따른 가중치를 적용한 Attention을 평균화**시키는 방법으로 이 방법은 **Multi-Head Attention** 효과와 대응합니다. 동일합니다.
- **Self-Attention**은 내부 Attention이라고도 부르며 **시퀀스 표현을 계산하기 위한 하나의 시퀀스**로 서로 다른 위치들과 관련된 Attention 구조입니다.



### Model Architecture

![image](https://user-images.githubusercontent.com/51338268/147523979-9fa2a119-ad8a-4322-9ecb-2fa829030843.png)

