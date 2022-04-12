---
title: Exploring the Data Efficiency of Cross-Lingual Post-Training in Pretrained Language Models
categories: [NLP]
comments: true
use_math: true
---

**Problem**

Even though language modeling is unsupervised and thus collecting data for it is relatively less expensive, it is still a challenging process for languages with limited resources

비록 언어 모델링이 비지도 학습 방법이고 그래서 비용이 상대적으로 덜 비싸지만, 여전히 제한된 자원의 언어들에게는 문제가 되는 과정입니다.

This results in great technological disparity between high- and low-resource languages for numerous downstream natural language processing tasks

그 결과 많은 자원과 적은 자원을 가진 언어들간에 NLP donwstream task들에 대한 엄청난 기술적 격차를 만들었습니다.

**Action(Contribution, Method, Solution)**

In this paper, we aim to make this technology more accessible by enabling data efficient training of pretrained   language models.

해당 논문에서 연구진은 사전학습된 언어모델을 사용 가능한 데이터로 효율적으로 학습함으로써 더욱 접근가능한 방법을 만드는데 초점을 두었습니다.

It is achieved by formulating language modeling of low-resource languages as a **domain adaptation task** using transformer-based language models pretrained on corpora of high-resource languages

적은 자원을 가진 언어의 언어 모델링을 하는데 많은 자원을 가진 언어 말뭉치로 사전학습이 되어진 transformer 기반의 언어 모델을 사용하여 **domain adaption task**를 함으로써 연구진은 실험(성취)할 수 있었습니다.

- Our novel cross-lingual post-training approach **selectively reuses parameters** of the language model trained on a high-resource language

  연구진의 novel cross-lingual post-training 접근법은 선택적으로 많은 자원의 언어로 학습되어진 언어모델의 **파라미터들을 선택적으로 재사용**합니다.

- **post-trains** them while learning language **specific parameters** in the low-resource language.

  학습된 언어의 특정 파라미터들은 저자원 언어로 사후 학습을 진행합니다.

- cross-lingual post-training (XPT)이 해당 방법론의 이름입니다.

  ![image](https://user-images.githubusercontent.com/51338268/162989545-eb7ca62f-5ccb-4635-9e98-d70c9e9bf807.png)

- 3.1. Transfer Learning as Post-Training

  - Post-training refers to the process of performing **additional unsupervised training to a PLM** such as BERT using unlabeled domain-specific data, prior to fine-tuning. It has been shown that this leads to improved performance by helping the PLM to adapt to the target domain

    사후-학습은 BERT와 같이 unlabeld domain-specific data를 사용한 **PLM에 추가적인 비지도 학습을 수행하는 과정을** 말하고 있으며, 차후에 fine-tuning을 해줍니다. 이 과정이 PLM을 target domain에 적응시키는데 도움을 줌으로써 성능의 향상을 이끌었다고 보고 있습니다.

  - Another key advantage of this approach is that this makes it possible to **completely skip the training in $L_S$**.

    해당 접근법의 또다른 중요 이점은 $**L_S$를 학습시키는 과정을 완전히 넘길 수 있게** 만들어줍니다.

  - This is because most recent publications in PLM literature make the trained model checkpoint publicly available, and the model architecture and training objectives in $L_S$ are inherited to $L_T$ when post-training.

    대부분의 공개된 PLM의 구조는 학습된 모델의 체크포인트를 공적 이용이 가능하도록 만들었고, $L_S$ 모델의 구조와 학습 목표는  $L_T$ 모델로 사후-학습이 되어질 때 상속되어집니다.

- 3.2. Selecting Parameters to Transfer

  - The most important part of the modeling process is the contextualization of embedding vectors, performed by the encoder layers

    모델링 과정에서 가장 중요한 부분은 encoder layers에 의해서 수행되어진 문맥의 의미를 담은 임베딩 벡터들입니다.

    We reuse them in post-training as these layers are known to acquire mostly language-independent knowledge

    연구진은 post-training에 encoder layer에서 가장 언어-독립 지식을 가진걸로 알려진 layer를 재사용하였습니다.

    It is possible to indirectly use them using bilingual word embedding techniques [26,45], but **we randomly initialize the word vectors of $L_T$ for simplicity.**

    소스 언어의 임베딩 벡터를 bilingual word embedding techniques로 간적접적으로 사용가능합니다, **하지만 연구진은 $L_T$의 단어 벡터들을 간결성을 위하여 무작위로 초기화시켰습니다.**

    We also reuse them in L_T as they are not language-dependent and have shown to improve performance in the preliminary experiments

    연구진은 또한 L_T를 언어 종속적인 않은 것도 활용하여 일반 연구에서 성능 향상을 보였습니다.

- 3.4. Two-Phase Post-Training

  - The parameters $E_{L_T}$, $θ_{ITL_{in}}$ , and $θ_{ITL_{out}}$ are randomly initialized and learned during the post-training phase. The noise introduced by this randomness can negatively impact the tuned parameters from $L_S$.

    $E_{L_T}$, $θ_{ITL_{in}}$ , and $θ_{ITL_{out}}$ 파라미터들은 무작위로 초기화고 되어지고 post-training 단계에서 학습되어집니다. 이러한 무작위성에 의한 노이즈들은 L_S로 파라미터들을 조정해주는데 부정적인 영향을 줄 수 있습니다.

  - To prevent this, we split the post-training into two phases, similar to **gradual unfreezing.**

    이러한 문제를 예방하기 위해서 연구진은 post-training을 2단계로 나누었으며 **점진적으로 unfreezing 시켜줍니다**.

    - In the first phase, **the parameters copied from the $L_S$ model are frozen, and only the $L_T$ embeddings and ITLs are learned** using the training examples in $L_T$.

      1단계에서는 $**L_S$모델의 파라미터들을 복사하고 동결시키며 오직 $L_T$ 임베딩 레이어, ITL 레이어만 $L_T$ 학습용 예제로 학습**시킵니다.

    - Phase two of our proposed method proceeds further and completely adapts the language model to $L_T$. **This is achieved by unfreezing the parameters from $L_S$ and finetuning the entire model using data in $L_T$.**

      2단계에서는 $L_T$ 언어모델로 완전히 적응시킵니다. $**L_S$에 있는 파라미터들을 해동을 시키고 $L_T$ 언어에 있는 모든 데이터들을 사용하여 fine-tuning을 수행합니다.**

We also propose **implicit translation layers** that can learn linguistic differences between languages at a sequence level

연구진은 또한 시퀀스 수준에서 언어들 사이의 언어적 차이점을 학습할 수 있는 **함축된 번역 레이어들**을 제안합니다.

- 3.3. Implicit Translation Layer

  - we propose the **Implicit Translation Layer** (ITL) to find this **mapping at a sequence level**.

    연구진은 **시퀀스 수준에서 맵핑관계**를 찾을 수 있는 **Implicit Translation Layer**를 제안합니다.

  - The ITL takes a sequence of vectors as input and outputs contextualized vectors of equal length

    ITL은 입력과 출력의 동일한 길이의 문맥적 의미를 담은 벡터들을 가지고 있습니다.

  - Two ITLs are added to the language model, one before the first encoder layer **(input-to-encoder)** and another one after the last encoder layer **(encoder-to-output)**.

    2개의 ITL은 언어 모델에 더해지는데, 하나는 인코더의 첫부분 **(입력 to 인코더)** 또 다른 하나는 인코더의 끝 부분 **(인코더 to 출력)** 에 더해줍니다.

**Result**

To evaluate our method, we post-train a RoBERTa model pretrained in English and conduct a case study for the **Korean language**.

연구진의 방법을 평가하기 위해서 연구진은 영어로 사전학습된 RoBERTa 모델을 사후학습을 시키고 **case-study로 한국어를 수행하였습니다.**

![image](https://user-images.githubusercontent.com/51338268/162989624-009577da-26ed-4758-a181-6188d8c20e6b.png)

![image](https://user-images.githubusercontent.com/51338268/162989707-4e7b5e7c-dd03-4003-bbf0-a152acef233f.png)

![image](https://user-images.githubusercontent.com/51338268/162989759-f74c4610-a1b1-4a1f-b8ff-7b9297e662ee.png)
