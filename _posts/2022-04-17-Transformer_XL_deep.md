---
title: Transformer-XL Paper
categories: [NLP]
comments: true
use_math: true
---



## Problem

Transformers have a potential of learning longer-term dependency, **but are limited by a fixed-length context** in the setting of language modeling.

Transformers는 더 긴 단어 종속성에 대한 잠재성을 가졌지만, 언어모델 설정에서의 **고정된 문맥이라는 한계를** 지니고 있습니다.



## Action(Contribution, Method)

We propose a novel neural architecture Transformer-XL that **enables learning dependency beyond a fixed length** without disrupting temporal coherence.

연구진은 기본 뉴런 구조인 Transformer-XL을 제안하며 해당 구조는 시간적 일관성을 방해하지 않고 **고정된 길이를 넘어서는 종속성을 학습시킬 수 있습니다**.

It consists of a **segment-level recurrence mechanism** and a **novel positional encoding scheme**.

**segment 수준의 재귀적 구조**와 기본 **postional encoding scheme**로 구성되어 있습니다.

- 2.1 **segment-level recurrence mechanism**

  - segment란? 하나의 token보다는 크고, 하나의 sequence보다는 작아 sequence를 임의의 길이로 나누어서 나온 문장의 길이.

    ![image](https://user-images.githubusercontent.com/51338268/163701520-780842ee-db3f-4f8a-a9cf-bc15e5eef5c2.png)

  - During training, the hidden state sequence computed for the previous segment is fixed and cached to be reused as an extended context when the model processes the next new segment.

    훈련하는 동안, 모델이 다음의 새로운 segment 처리할 때, 은닉 시퀀스는 확장된 context에 재사용되기 위해서 일시적으로 기록되어지고 고정된 이전의 segment를 계산합니다.

  - Although the **gradient still remains within a segment**, this additional input **allows the network to exploit information in the history**, leading to an ability of modeling **longer-term dependency** and **avoiding context fragmentation**.

    비록 **그래디언트가 segment안에 여전히 남더라도**, 이러한 추가적인 입력값은 **네트워크가 과거의 정보를 활용하도록 허락**하게됩니다, 이는 모델이 **더 긴 종속성**과 **문맥 단절 문제를 피하는 능력**을 이끌어 줍니다.

  - $s_τ = [x_{τ,1}, · · · , x_{τ,L}]$ and $s_{τ+1} = [x_{τ+1,1}, · · · , x_{τ+1,L}]$ respectively. Denoting the n-th layer hidden state sequence produced for the τ -th segment $s_τ$ by $h^n_τ ∈ R^{L×d},$ where d is the hidden dimension. Then, the n-th layer hidden state for segment $s_{τ+1}$ is produced (schematically) as follows,

    n번째 hidden state sequence는 τ 번째 $s_τ$을 L x d 차원의 실수공간에서 포함되며 d는 hidden dimension입니다. 그리고, $n$번째 층의 hidden state에서 $s_{τ+1}$의 식은 다음과 같습니다.

    ![image](https://user-images.githubusercontent.com/51338268/163701530-4165cf71-5b89-48de-b92e-0ed78df0f550.png)

  - where the function $SG(·)$ stands for stop-gradient, the notation $[h_u ◦ h_v]$ indicates the concatenation of two hidden sequences along the length dimension, and W·

    $SG(·)$의 의미는 그래디언트를 멈추라는 의미이고 $[h_u ◦ h_v]$는 2개의 hidden sequence를 concatenation하라는 의미입니다.

- 2.2 **postional encdoing scheme(Relative Positional Encodings)**

  - how can we keep the positional information coherent when we reuse the states?

    어떻게 하면 state를 재사용할 때, 위치 정보의 일관성을 유지시킬 수 있을까?

    ![image](https://user-images.githubusercontent.com/51338268/163702452-bf61909b-2079-473c-922c-7349fbf1bc7d.png)

    기본 Transformer의 attention score를 구하는 방식은 query와 key를 내적한 형태로 위의 식에서 E_{x_i}는 i번째 워드 임베딩 벡터이고 U_i는 Positional Encoding을 의미합니다.

    해당 식은 전개하면 아래와 같은 식이 나옵니다.

    ![image](https://user-images.githubusercontent.com/51338268/163702478-6c6fda2f-d522-4116-a017-922b3f9322af.png)

    Transformer-XL의 경우 다음과 같이 변합니다.

    ![image](https://user-images.githubusercontent.com/51338268/163702484-c0629218-f482-426a-b043-68cee21c9b8f.png)

  - The first change we make is to **replace all appearances of the absolute positional embedding $U_j$ for computing key vectors in term (b) and (d) with its relative counterpart $R_{i−j}$** . This essentially reflects the prior that only the relative distance matters for where to attend. Note that R is a sinusoid encoding matrix (Vaswani et al.,2017) without learnable parameters.

    첫 번째 변화는 (b)와 (d)에 있는 **absolute positional embedding인 $U_j$를 key vector들의 상대적인 짝인 $R_{i-j}$로 바꿨습니다.** 이는 필수적으로 어디에 주목하든 상대적인 거리만을 담은 사전 정보를 반영케 합니다. R은 학습가능하지 않은 파라미터인 sinusoid encoding matrix입니다.

  - Secondly, we introduce a trainable parameter $u ∈ R^d$ to replace the query $U^T_i W^T_q$ in term (c). In this case, **since the query vector is the same for all query positions**, it suggests that the attentive bias towards different words should remain the same regardless of the query position. With a similar reasoning, a trainable parameter $v ∈ R$ d is added to substitute $U^T_i W^T_q$ in term (d).

    두 번쨰로, (c)에서 연구진은 $U^T_i W^T_q$를 학습 가능한 파라미터인 $u$로 대체합니다.

    **이 경우 query vector는 모든 query position에 대해서 같기 때문에,** 다른 단어로 향하는 attentive bias들은 query position과 동일한 것을 남길 수 있습니다. 이와 같은 이유로 (d)에 있는 $U^T_i W^T_q$도 $v$로 대체시킵니다.

    ![image](https://user-images.githubusercontent.com/51338268/163702499-93d8c880-aca5-48aa-9a61-56093e23e19b.png)

  - Finally, we deliberately separate the two weight matrices $W_{k.E}$ and $W_{k.R}$ for producing the content-based key vectors and location-based key vectors respectively.

    마지막으로, 연구진은 $W_{k.E}$와 $W_{k.R}$로 나눌 수 있고 문장 기반 key vector와 위치 기반 key vector로 나눌 수 있게 됩니다.

Our method not only enables capturing longer-term dependency, but also resolves the context fragmentation problem.

연구진의 방법은 더 길어진 단어 종속성 뿐만 아니라, context fragmentation problem도 해결하였습니다.



## Result(Experiment)

As a result, TransformerXL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers, achieves better performance on both short and long sequences, and is up to 1,800+ times faster than vanilla Transformers during evaluation.

결과적으로, Transforemr-XL은 RNN보다 80% 긴 문장을 학습하고 기본 Transformers보다 450% 더 길게 학습을 하면서, 짧고 긴 시퀀스에서도 더 좋은 성능을 성취하였고, 기본 Transformers보다 평가 시간에 1,800초 이상 더 빨랐습니다.

Notably, we improve the state-of the-art results of bpc/perplexity to 0.99 on enwiki8, 1.08 on text8, 18.3 on WikiText-103, 21.8 on One Billion Word, and 54.5 on Penn Treebank (without finetuning).

정확하게는, 연구진은 enwiki8에서 perplexity 0.99로 SOTA의 결과를 향상시켰습니다.(fine-tuning없이)

When trained only on WikiText-103, Transformer-XL manages to generate reasonably coherent, novel text articles with thousands of tokens. Our code, pretrained models, and hyperparameters are available in both Tensorflow and PyTorch1.

WikiText-103을 학습시킬 때만, Trnasformer-XL은 수천개의 토큰을 가진 기본 텍스트 기사들에서 합리적인 일관성을 생성해내었습니다. 연구진의 사전학습된 코드와 모델 하이퍼파라미터는 tensorflow와 pytorch 모두 사용 가능합니다.



## Reference

- https://medium.com/@serotoninpm/논문-nlp-transformer-xl-c0926e9b51a4
- https://medium.com/dair-ai/a-light-introduction-to-transformer-xl-be5737feb13
- https://ai.googleblog.com/2019/01/transformer-xl-unleashing-potential-of.html
- https://baekyeongmin.github.io/paper-review/transformer-xl-review/
