---
title: NLP Paper (09) - BERT
categories: [NLP]
comments: true
use_math: true
---



# [BERT] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding



**[BERT] BERT: 언어 이해를 위한 깊은 양방향 Transformer 사전 학습 시키기**



해당 번역본은 정식 번역본이 아니며 개인이 공부를 위해서 번역한 내용으로 많은 오역, 오타가 존재합니다. 이러한 점을 감안해주시고 읽어주시면 감사하겠습니다.



## Abstract

- 해당 논문은 BERT라는 새로운 언어 이해 모델을 소개합니다.
- BERT는 Bidirectional Encoder Representation from Transformers의 약자로 해석하면 'Transformers로부터온 양방향 인코더 표현' 입니다.
- BERT는 **레이블 되지 않은 텍스트를 문장의 왼쪽, 오른쪽을 전 층에서 동시에 적용함으로써 양방향 표현을 사전 학습하도록 설계**되었습니다.
- 사전 학습되어진 BERT 모델은 **단지 하나의 추가적인 출력층을 미세조정(Fine-tune)**하여 광범위한 분야의 태스크에서 SOTA 수준의 모델들을 만들 수 있습니다.



### Introduction

- 최근들어 언어 모델을 사전학습시키는 것은 많은 NLP 태스크의 성능을 효과적으로 향상시켜 왔습니다.
- 사전 학습된 언어 표현을 다운 스트림 태스크에 적용하는 전략으로 2가지가 존재합니다 : Feature-based 와 Fine-tuning
- Feature-based의 예시로는 ELMo가 존재하며 태스크에 따른 특수한 구조를 사전학습된 표현에 추가적인 표현을 사용합니다.
- Fine-tuning은 대표적으로 GPT가 존재하며  최소한의 작업 파라미터를 사용하고 다운 스트림 태스크를 간단히 모든 사전 학습된 파라미터들을 fine-tuning하여 학습되어집니다.
- 위의 두 방식은 단방향으로 일반화된 언어 표현을 학습하는데, 이 방식은 사전 학습된 표현들의 능력을 제한하는 기술이라고 여기고 있습니다.
- 해당 논문에서는 **Fine-tuning 기반의 접근법이며 단방향성의 제약을 '마스크되어진 언어 모델'(MLM)**을 사용하여 완화시키려합니다.
- **마스크되어진 언어 모델을 입력의 토큰들중에서 무작위로 마스크를 씌어주는 방식**으로 모델은 마스크된 단어의 단어장 id를 오직 그 문맥 자체를 기반으로 예측하게 됩니다.
- 마스크된 언어 모델은 또한 다음 문장 예측 태스크에도 사용이 가능합니다.



