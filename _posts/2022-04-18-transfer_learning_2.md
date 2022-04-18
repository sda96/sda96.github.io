---
title: Cross-lingual Transfer Learning for Japanese Named Entity Recognition
categories: [NLP]
comments: true
use_math: true
---



## 1. Problem

Due to the growing interest in voice-controlled devices, such as Amazon Alexa-enabled devices or Google Home, porting these devices to new languages quickly and cheaply has become an important goal.

Amazon의 Alexa나 Google Home과 같은 음성 인식 장비들에 대한 열기가 높아짐에 따라, 이러한 장비들은 새로운 언어로 빠르고 값싸게 이전시키는 것이 주요한 목표입니다.

One of the main components of such a device is a model for Named Entity Recognition (NER).

이러한 장비들의 주요한 요소중에 하나는 Named Entity Recognition(NER) 모델입니다.

Typically, NER models are trained on large amounts of annotated training data.

일반적으로, NER 모델들은 엄청난 양의 주석처리 되어진 훈련 데이터로 훈련되어집니다.

However, collecting and annotating the required data to bootstrap a large-scale NER model for an industry application with **reasonable performance is time-consuming, costly**, and it doesn’t scale to a growing number of new languages.

하지만, 거대한 크기의 NER 모델을 만들기 위해서 요구된 데이터들을 **상용화에 합리적인 성능을 보일 정도로 주석 처리를 하려면 시간이 많이 소비되고, 비용도 들며** 그러면 새로운 언어들의 커다란 성장은 하기 어려워집니다.

Aiming to reduce the time and costs needed for bootstrapping an NER model for a new language, we leverage existing resources.

시간과 비용을 줄이는데 초점을 맞추면 새로운 언어에 대한 NER 모델을 부트스트랩핑할 필요가 있는데, 연구진은 존재하는 자원들을 활용하고자 합니다.

In particular, we explore cross-lingual transfer learning, in which weights from a trained model in the source language are transferred to a model in the target language.

특히, 연구진은 cross-lingual transfer learning을 활용코자 하며, 이 방법은 source 언어로 학습되어진 모델의 weight들을 target 언어를 쓰는 모델로 전이시키는 방법입니다.

Transfer learning (TL) has been shown previously to improve performance for target models (Yang et al., 2017; Lee et al., 2017; Riedl and Pado´, 2018).

Transfer Learning은 이전에도 taget 모델들의 성능 향상을 보여주고 있습니다.

However, work related to cross-lingual transfer learning for NER has mainly focused on rather similar languages, e.g. transferring from English to German or Spanish.

하지만 NER을 위한 cross-lingual transfer learning과 연관된 작업들은 주로 영어에서 독일어, 스페인어와 같은 비슷한 언어로 전이시키는데 집중되어 있습니다.

**In contrast, we focus on transferring between dissimilar languages, i.e. from English to Japanese.**

**대조적으로, 연구진은 유사하지 않은 언어인 영어에서 일본어로 전이시키는데 집중하고 있습니다.**

We present experimental results on external, i.e. publicly available, corpora, as well as on internally gathered large-scale real-world datasets.

연구진은 내부적으로 모은 large-scale real-world 데이터셋 뿐만 아니라, 공적으로 이용 가능한 말뭉치들로 연구한 결과를 발표하고 있습니다.



## 2. Action

First, a deep neural network model is developed for NER, and we extensively explore which combinations of weights are most useful for transferring information from English to Japanese.

첫 번째로, 딥 뉴럴 네트워크 모델은 NER를 위해서 발전되었고, 연구진은 영어에서 일본어로 정보를 전이시키기 위해서 가장 유용한 weight들의 조합을 넓게 탐색하였습니다.

Furthermore, aiming to overcome the linguistic and orthographic dissimilarity between English and Japanese, **we propose to romanize the Japanese input, i.e. convert the Japanese text into the Latin alphabet.**

게다가, 언어적, 직교적으로 영어와 일본어간의 비유사성을 극복하는데 집중하여, **연구진은 일본어의 입력을 로마자표기로 전환하는 것을 제안합니다. i.e 일본어 텍스트를 로마자 알파벳으로 바꿈**

This results in a common character embedding space between the two languages, and intuitively should allow for more efficient transfer learning at the character level.

해당 결과는 두 언어 사이에 동일한 문자 임베딩 공간에 존재하며 직관적으로 문자 단위에서 더 효율적인 Transfer Learning을 할 수 있을 것으로 보입니다.



### 2.1 Architecture

![image](https://user-images.githubusercontent.com/51338268/163765970-6398c595-8542-4244-8c14-b832bf787130.png)

For our baseline NER system **we use a BiLSTM architecture that takes word and character embeddings as input.**

연구진의 NER 모델의 베이스라인은 **BiLSTM 구조로 입력으로 워드 임베딩과 문자 임베딩 2가지가 들어갑니다.**

**The same architecture is used both for the source and the target languages** to allow for transfer of weights when the cross-lingual TL is applied.

cross-lingual TL이 적용되어질 때, weight들을 target 언어로 전이시키기 위해서, 사용되는 **source와 target 모델들은 모두 동일한 구조**가 사용되어집니다.

This architecture largely resembles the model in Lample et al. (2016), except for the final CRF layer.

해당 구조는 넓게 보았을 때, 마지막 CRF 층을 제외한 Lample 에 있는 모델을 조립한 것 입니다.

For every token, word and character embeddings are generated.

모든 토큰은 단어, 문자 임베딩에 의해서 생성되어 집니다.

The latter are passed through a character Bi-LSTM, the output of which is concatenated with the word embeddings.

문자가 문자 BiLSTM을 지나가면, 해당 출력값은 단어 임베딩과 concat 되어집니다.

This combined representation is then passed into the word Bi-LSTM, followed by a dense layer and a final softmax layer.

앞서 조합된 representation은 그리고나서 단어 Bi-LSTM을 지나가고 따라서 dense layer를 지나 마지막 softmax layer를 통과합니다.

An example for English is presented in Figure 1. **Note that the character level inputs in this figure are unigrams, but in practice we use bigrams, i.e. “Ye” and “es” for “Yes”.**

영어에 대한 예시는 그림1과 같으며, 그림에서는 **입력 문자 단위가 unigram이지만 연구진은 실제로 bigram을 사용했습니다.**



### 2.2 Proposed model

- We group our weights together as shown in Figure 1 (grouped layers in boxes): **character embeddings and character Bi-LSTM weights form the “character weights”, word embeddings and word Bi-LSTM weights form the “word weights”, and dense layer weights form the “dense weights”.**

  연구진은 그림1에 있는 weight들을 그룹화시켰으며 각 그룹은 다음과 같습니다

  - **문자 임베딩과 문자 Bi-LSTM 가중치들은 “character weights”**
  - **단어 임베딩과 단어 Bi-LSTM 가중치들은 “word weights”**
  - **dense layer 가중치들은 “dense weights”**

  when transferring embeddings, we only update the vectors that correspond to char n-grams or words observed in both the source and target training data.

  임베딩을 전이시킬 때, 연구진은 오직 문자 n-gram 혹은 훈련 데이터에 source와 target에 동시에 관측된 단어들만 벡터들을 업데이트 시킵니다.

- **For transferring to a target language with a different writing system than the source one we propose the Mixed Orthographic Model (MOM).**

  **source와 target언어가 전이를 시키는 데 서로 다른 쓰기 시스템을 가지는 경우 연구진은 Mixed Orthographic Model(MOM)을 제안합니다.**

  **Specifically, the character layer inputs are romanized while the word layer inputs are kept in their original Japanese text.**

  **특히, word layer 입력들은 원본 일본어 텍스트와 동일한 동안 character layer 입력들은  로마자로 표기되어집니다.**

  This allows for transfer of character information from a source to a target language with originally different writing systems by creating a common and overlapping character embedding space.

  이 방법은 공통되고 겹쳐지는 문자 임베딩 공간을 생성함으로써, 원래는 서로 다른 쓰기 시스템을 가진 두 언어를 source에서 target 언어로 문자 정보를 전이시키는데 허락해줍니다.

  At the same time, keeping the original Japanese text in the word level allows us to keep the capacity to disambiguate homophones, which is lost via the romanizing process as explained in the previous section (Section 3.2).

  동시에, 단어 수준의 원본 일본어를 유지시키는 것은 동음이의어를 명확하게 해주는 능력을 유지하도록 해주며, 이는 앞서 설명한 로마자 표기 처리방법에의해서 읽게 됩니다.

  ![image](https://user-images.githubusercontent.com/51338268/163766012-084ee1fa-1ba7-48ba-b330-81d08133ae33.png)

  **word input은 일본어, char input은 로마자 표기**



## 3. Result

Gains with TL are achieved on all evaluated target datasets, even large-scale industrial ones.

TL로 얻은 결과는 평가용 타겟 데이터셋과 심지어는 거대 산업 데이터셋에도 적용해봤으며. Moreover, the effect of TL on the target dataset size and of the target tagset distribution is investigated.

게다가 TL은 target dataset의 크기와 target tageset 분포에 영향을 받습니다.

Finally, we show that similar gains are achieved when applying the proposed approach from English to German, indicating the possibility to generalize it both to European and non-European target languages.

마지막으로, 연구진은 영어에서 독일어로 전이될 때, 제안되어진 접근법을 적용한 것과 비슷한 성과를 보였으며, 이 말은 유럽관과 비유럽권 target 언어들이 일반화되어졌을 가능성을 제시합니다.



### 3.1 Experiment

For our experiments we make use of datasets in three languages. First, **an English dataset is used to train the source NER model**.

해당 논문의 실험에서 3가지 언어가 있는 데이터셋을 사용하였으며, **첫 번째로 영어 데이터셋은 source NER 모델을 학습시키는데 사용되었습니다**.

Then, a target language dataset, which is smaller in size than the source dataset, is used to build a **target NER model**. This serves as the target baseline.

그리고 나서 target 언어 데이터셋은 soruce 언어 데이터셋보다 크기가 작으며 **target NER model**로 만들어졌으며 이것이 target 베이스라인으로서 제공됩니다.

We evaluate our approach both on **external and internal datasets**.

연구진은 **external 와 internal 데이터셋**으로 해당 접근법을 평가하였습니다.

External datasets are composed of company data and are mainly used for comparing our monolingual models to the state-of-the-art, while internal datasets are composed of publicly available data and are used to explore potential data reductions in a real-world large-scale industry setting.

External dataset은 회사 데이터와 주로 monlingual model의 SOTA를 비교할 때 사용되는 데이터로 구성되어 있으며, 반면아 Internal 데이터셋은 공적으로 이용 가능한 데이터와 현실 세계에 사용될 대규모 산업 세팅의 데이터를 사용하였습니다.

![image](https://user-images.githubusercontent.com/51338268/163766097-153bc342-91fe-4884-b670-a47a814f6a51.png)

- **Layer combinations for TL**

  - Section 3.3 : Char Weight, Word Weight, Dense Weight

  - The layer groups defined in Section 3.3 are combined and experiments are conducted on the two external JP datasets as well as on a subset of the JP “Medium” internal one

    layer 그룹들은 앞서 정의된 경우며 이들을 조합한 실험결과는 다음과 같고 수행한 실험은 영어 source 모델에서 2개의 external JP 데이터셋과 Internal JP 데이터셋 사용한 일본어 target 모델로 Transfer Learning을 시켰을 때 입니다.

    ![image](https://user-images.githubusercontent.com/51338268/163766171-c35b836e-e7b1-438e-b0e9-6f06fb0e4ef1.png)

- **Effect of romanization of Japanese on TL**

  ![image](https://user-images.githubusercontent.com/51338268/163766232-a1c0079f-3ce0-48a3-b82e-555c0ef7dc20.png)



## 4. Reference

- https://aclanthology.org/N19-2023.pdf
