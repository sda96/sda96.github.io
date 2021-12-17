---
title: 머신러닝 모델들 간략 정리
categories: [machinelearning]
comments: true
use_math: true
---



# [SentencePiece] SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing Units



해당 내용들은 github에 정리한 내용을 글씨만 가져왔으며 머신러닝 모델 패키지인 sklearn을 사용한 코드들은 해당 [링크](https://github.com/sda96/Going_Deeper_Project/blob/main/04_New_multiclassification/02_Going_Deeper_Project_5000.ipynb)에서 보실 수 있습니다.



### 1. [MultinomialNB 모델](https://www.youtube.com/watch?v=3JWLIV3NaoQ)

- 교집합 : $P(A \cap B) = P(A , B) = P(A \lvert B) \cdot P(B)$  
    - 두 사건 $A, B$가 서로 독립인 경우 $P(A \cap B) = P(A)\cdot P(B)$ 
    - 두 사건 $A, B$가 서로 종속인 경우 $P(A \cap B) = P(A \lvert B) \cdot P(B)$

- 조건부 확률
    - $P(A \lvert B) = \frac{P(A \cap B)}{P(B)}$  
- 베이즈 정리
    - $P(A), P(B), P(B \lvert A)$를 알고 있을 때, $P(A \lvert B)$를 구할 수 있습니다.
    - $P(A \lvert B) = \frac{P(B \lvert A) \cdot P(A)}{P(B)}$
    - 예를 들어 스팸 메일을 분리하면서 "free"라는 단어가 자주 보여 해당 단어가 포함된 메일은 스팸으로 분리하려고 합니다.  
      이때, "free"라는 단어가 포함하였는데 스팸메일로 분류되는 확률을 구하겠습니다.
        - 사건 $A$가 전체 메일 중에서 "free" 라는 단어가 포함되어 있는 경우
        - 사건 $B$가 전체 메일 중에서 스팸메일인 경우  
        - $P(B|A) = \frac{P(B) \cdot P(A \lvert B)}{P(A)}$
        - $posterior = \frac{prior \times likelihood}{evidence}$

해당 모델을 현재의 데이터셋에 적용한 경우 label의 클래스들을 $B_i$, featrues에 있는 단어들을 각각 $A_1, A_2 ... A_n$라고 하겠습니다.

이때의 식은 $P(B_i \lvert A_1, A_2 .... A_n)$로 정리가 가능하며 계산한 결과 가장 확률이 높게 나온 i번째 클래스로 분류를 합니다.

**참고 사이트**

- [집합의 연산중 결합법칙을 발견한 사이트](https://deep-learning-study.tistory.com/419)  
- [나이브 베이즈 위키 백과](https://ko.wikipedia.org/wiki/%EB%82%98%EC%9D%B4%EB%B8%8C_%EB%B2%A0%EC%9D%B4%EC%A6%88_%EB%B6%84%EB%A5%98)



### 2. ComplementNB 모델

ComplementNB 모델은 기존의 NB 모델에서 데이터의 label 불균형 문제를 개선시킨 모델로 각 label마다 가중치를 부여하여 불균형 문제를 개선한 모델입니다.



### 3. Logistic Regression 모델

로지스틱 회귀모델에 대한 정리는 해당 [링크](https://sda96.github.io/2021-10/classification_problem)에 정리하였습니다. 하지만 일반적인 로지스틱 회귀 모델은 이진 분류 모델이라고 알고 있는데 어떻게 다중 분류 문제를 해결할 수 있는지 알아보았습니다.

sklearn 패키지의 로지스틱 회귀 모델은 다중 분류 문제가 들어오는 경우 OvR(One-vs-rest) 알고리즘을 사용합니다.
- [sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)  
- [다중 분류 알고리즘 OvR](https://brunch.co.kr/@linecard/482)

OvR 알고리즘은 만약 파랑, 검정, 빨강이라는 클래스에서 파랑을 분류하는 경우 파랑 클래스의 데이터를 제외하고는 모두 똑같은 클래스라고 판단하여 결국 '파랑 클래스냐, 파랑 클래스가 아니냐.' 라는 이진 분류 문제로 바꾸어서 생각하며 이 방법을 파랑, 검정, 빨강에 모두 적용하여 결국에는 다중 분류 문제를 해결할 수 있게 됩니다.



### 4. 선형 서포트 벡터 분류기

[![image](https://user-images.githubusercontent.com/51338268/146365205-69972aa6-98d5-45f3-95d4-a00cbe313ab4.png)](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=gdpresent&logNo=221717231990)


선형 서포트 벡터 분류기은 그림과 같이 데이터들을 2개의 클래스로 분류하는 결정경계를 찾으며 **단, Margin이 최대가 되는 결정경계를 찾아서 이진 분류를 시키는 알고리즘**입니다.

결정경계는 두 클래스를 분류하는 선이 되며, Support Vectors는 각 클래스 별로 Margin에 가장 가까운 데이터를 의미합니다. 서포트 벡터 머신이 학습하는 과정에서 사용되는 데이터가 2개 뿐이기에 계산량이 상당히 적게 소모한다는 것을 알 수가 있습니다.

하지만 종종 선형 구분이 되지 않는 문제가 발생하며 이 문제를 해결하기 위해서 더 높은 차원으로 매핑하여 분리하는 방법이 제안되었으며 해당 과정에서 늘어나는 계산량을 막기 위해서 커널함수를 정의한 SVM(Support Vector Machine) 구조를 설계하였습니다.
- [SVM 위키 백과 설명](https://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0)
- [SVM 설명](https://blog.naver.com/tjdudwo93/221051481147)

SVM의 파라미터로 cost와 gamma가 존재합니다
- cost : 결정경계의 margin의 간격을 결정하며 cost가 작으면 margin은 넓어지고 cost가 크면 margin은 좁아집니다.
- gamma : train 데이터 하나당 영향을 끼치는 범위를 의미합니다.



### 5. [의사결정나무](https://www.youtube.com/watch?v=n0p0120Gxqk)

![image](https://user-images.githubusercontent.com/51338268/146504553-907c454d-ff00-404d-83c8-e6a5cf8439fd.png)


의사결정나무는 특성, 변수마다 이진분류를 진행하여 마지막 끝노드에 있는 비율이 높은 클래스로 분류하는 방식의 분류 모델이기도 하고, 끝노드의 데이터들의 평균이나 중앙값을 사용한 예측 모델이기도 합니다.

의사결정나무의 가장 기본적인 알고리즘인 ID3 알고리즘은 2가지 개념을 토대로 분류를 진행합니다.
- Entropy가 작아지는 방향으로 알고리즘이 진행됨
    - Entropy는 정보량의 기대값을 의미합니다.
    - $H[Y] = -\sum^{n}_{i=1} p_i \log p_i$
- Information gain이 가장 높은 특성부터 분류를 진행함
    - information gain은 X라는 조건에 의해 확률 변수 Y의 엔트로피가 얼마나 감소하였는지를 나태나는 값입니다.
    - $IG[Y, x] = H[Y] - H[Y \lvert X]$

분류는 정해진 Entropy의 수치나 나누어지는 속성을 모두 사용하거나 정해진 속성의 개수만큼 사용하면 멈추게 됩니다.

참고사이트
- [의사결정나무](https://datascienceschool.net/03%20machine%20learning/12.01%20%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4.html)



### 6. [랜덤포레스트](https://www.youtube.com/watch?v=nZB37IBCiSA)

[![image](https://user-images.githubusercontent.com/51338268/146507957-8be96281-3b09-4f0e-9563-2ec66261948f.png)](https://www.tibco.com/reference-center/what-is-a-random-forest)
랜덤포레스트는 기존의 의사결정나무를 기반으로 만들어진 알고리즘으로 다수의 의사결정나무 모델을 사용하여 해당 모델들의 결과를 투표하는 방식으로 데이터를 분류합니다.

랜덤포레스트의 특성은 다음과 같습니다.
- boosting
    - boosting은 각각의 의사결정 나무 모델에 일부러 반복추출된 데이터를 넣어서 해당 데이터에 과대적합되어 편향된 모델들을 만듭니다.
    - 특정 데이터에 편향됨으로써 그 데이터에 편향된 모델을 그 데이터에 대한 성능이 더 좋게 나올거라 기대합니다.
- random selection
    - 기존의 의사결정나무는 가장 좋은, information gain이 가장 높은 속성부터 분류를 하였습니다.
    - 하지만 랜덤포레스트는 무작위 속성을 선택하여 분류합니다.
- aggregating
    - 다수의 의사결정나무 모델로부터 다양한 결과가 나왔을 텐데 그 결과들을 토대로 투표를 진행하여 가장 많이 나온 클래스로 분류합니다.
    

흔히 boosting과 aggregating을 더한 기법은 bagging이라고 부릅니다.

랜덤포레스트는 각 변수들이 얼마만큼 종속변수에 영향을 주는가에 대한 측정이 가능하며 계산하는 방법은 다음 [링크](https://velog.io/@vvakki_/%EB%9E%9C%EB%8D%A4-%ED%8F%AC%EB%A0%88%EC%8A%A4%ED%8A%B8%EC%97%90%EC%84%9C%EC%9D%98-%EB%B3%80%EC%88%98-%EC%A4%91%EC%9A%94%EB%8F%84Variable-Importance-3%EA%B0%80%EC%A7%80)에 있습니다.



### 7. [그래디언트 부스팅](https://3months.tistory.com/368)

- 부스팅은 약한 분류기, 모델을 결합하는데 일반적으로 의사결정나무를 사용하며 하나의 강한 분류기를 만드는 과정을 말합니다.

- 그래디언트는 해당 앙상블 모델의 예측값과 실제값의 차이인 잔차, 손실함수를 줄여나가는 방향으로 학습을 진행합니다.

- 예측 모델의 경우 손실함수로 RMSE를 사용할 수 있지만 분류 모델에서는 이진 교차 엔트로피나 교차 엔트로피를 사용할 수 있습니다.

**참고사이트**

- [그래디언트 부스팅](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-15-Gradient-Boost)



### 8. [보팅](https://www.youtube.com/watch?v=y4Wh0E1d4oE&feature=youtu.be)

앙상블 기법이란 여러개의 모델들의 결과들을 혼합하여 더 개선된 결과를 만드는 모델을 만드는 방법을 말합니다.

보팅은 앙상블 기법중에 하나로 Soft Voting과 Hard Voting이 존재합니다.
- Hard Voting은 일반적인 다수결의 원칙을 따르며 각 모델이 예측한 결과중에서 가장 빈도가 높은 클래스를 결과로 반환하는 방식입니다.
[![image](https://user-images.githubusercontent.com/51338268/146578648-c30faa62-2ef8-435e-9492-80e70421651d.png)](https://stats.stackexchange.com/questions/349540/hard-voting-soft-voting-in-ensemble-based-methods)
- Soft Voting은 각 모델마다 클래스별 확률값을 가지는 즉, 클래스에 따른 확률분포를 가지는데 모델마다 가지는 클래스별 확률값들을 평균을 내서 가장 높은 클래스를 결과로 반환하는 방식입니다.
[![image](https://user-images.githubusercontent.com/51338268/146581462-91b63239-3943-493a-ab00-b15ce9e599ca.png)](https://medium.com/wids-mysore/ensemble-learning-techniques-votingclassifier-c4b38ee62129)
