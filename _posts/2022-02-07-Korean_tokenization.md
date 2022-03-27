---
title: 네이버 HyperCLOVA 토큰화 파헤치기
categories: [NLP]
comments: true
use_math: true
---



## 1. HyperCLOVA 토큰화 방법

한국어는 영어와 다르게 교착어의 특성상 빈칸 단위로 문장을 토큰화를 시키면 의미가 이어지지 않는 문제가 발생하게 됩니다.

이를 문제를 해결하기 위해서 한국어에서 의미를 담은 가장 작은 단위인 형태소(morpheme) 단위로 토큰화를 시키면 문제를 해결할 수 있으며 일반적으로 자주 사용하는 형태소 분석기는 Mecab-ko 입니다.

하지만 GPT-2와 GPT-3와 같은 모델들의 토큰화는 byte-level BPE를 사용하고 있는데 한국어를 byte 수준에서 토큰화를 시키게 되면 한국어 발음 체계를 고려하지 않게 되어서  'ㅎ', '하', '한' 이 모두 다른 byte로 인식되게 됩니다.

그래서 HyperCLOVA 에서는 morpheme-aware-byte-level BPE를 적용하였으며 이 방법으로 총 5618억개의 토큰을 사용하였습니다.



## 2. Morpheme-aware-byte-level BPE

| Tokenization               | Description                                                  | Tokenized Sequence                          |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------- |
| Raw Text                   | 원본                                                         | 나랑 쇼핑하자                               |
| Consonant and Vowel (CV)   | 자모 단위 토큰화                                             | ㄴ/ㅏ/ㄹ/ㅏ/ㅇ/*/ㅅ/ㅛ/ㅍ/ㅣ/ㅐ/ㅎ/ㅏ/ㅈ/ㅏ |
| Syllable                   | 음절 단위 토큰화                                             | 나/랑/*쇼/핑/하/자                          |
| Morpheme                   | 형태소 단위 토큰화                                           | 나/랑/*/쇼핑/하/자                          |
| Subword                    | SentencePiece에 있는 BPE                                     | _나랑/\_쇼/핑하/자/                         |
| **Morpheme-aware Subword** | **형태소 단위 토큰화 적용후 <br /> 형태소 단위로 분절된 토큰에 BPE 적용** | **_나/\_랑/*/\_쇼/핑/\_하/\_자**            |
| Word                       | 공백 단위로 토큰화                                           | 나랑/쇼핑하자                               |

위의 테이블은 An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks 논문에서 실험한 5가지 토큰화 기법에 대한 내용으로 각 토큰화 기법을 바탕으로 성능을 비교하였으며 성능의 지표는 한영번역 모델의 BLEU-score를 기준으로 하였습니다.

![image](https://user-images.githubusercontent.com/51338268/160271240-1b9d51f5-951c-4a07-a968-0c6cf159176b.png)

성능 비교 결과 Morpheme-aware Subword에서 vocab size가 32K인 경우가 가장 좋은 성능을 선보였으며 특이한 점은 vocab size가 64K일 때는 오히려 성능이 떨어지는 것을 볼 수 가 있습니다.

![image](https://user-images.githubusercontent.com/51338268/160272459-f4636fe2-df2a-48ae-9155-75a86270e2f5.png)

번역 task 말고도 다른 NLU task에서 성능을 비교한 결과 예외적으로 Korquad에서만 Subword의 성능이 더 뛰어나고 나머지 task에서는 Morpheme-aware Subword 방식이 가장 뛰어난 것으로 나타났으며 앞선 2가지 table을 통하여 vocabulary size가 모델의 성능을 항상 보장하는 것은 아니라는 점을 알 수 있었습니다.



## 3. Morpheme-aware-byte-level BPE 구현

![image](https://user-images.githubusercontent.com/51338268/160272590-dcc4e5d8-cc7c-44a4-ae99-4a5b932a80ff.png)

An Empirical Study of Tokenization Strategies for Various Korean NLP Tasks 논문의 공식 Github repository에 있는 코드로 해당 코드를 통하여 입력으로 들어오는 문장을 2가지 단계로 나누어서 처리됩니다.

1. Mecab-ko를 활용한 형태소 단위 토큰으로 이루진 리스트
2. ```join()```함수로 리스트를 빈칸 단위로 합쳐준 뒤, BPE 적용



## Reference

- [HyperCLOVA 논문](https://arxiv.org/pdf/2109.04650.pdf)
- [HyperCLOVA 분석 참고 사이트](https://jiho-ml.com/weekly-nlp-45/)
- [Tokenization Strategies](https://arxiv.org/pdf/2010.02534.pdf)
- [Tokenization Strategies 논문 리뷰](https://cpm0722.github.io/paper-review/an-empirical-study-of-tokenization-strategies-for-various-korean-nlp-tasks)

- [Morpheme-aware-byte-level BPE 코드](https://github.com/kakaobrain/kortok/blob/main/tokenizer/mecab_sp.py#L19)
