---
title: 비지도 학습 (1) 군집화
categories: [deeplearning]
comments: true
---



## 1. 지도학습과 비지도 학습

딥러닝 모델들로 다양한 유형의 문제들을 해결이 가능한데 문제의 종류에 따라서 해결 가능한 학습방법들이 조금씩 다르게 존재합니다.

지도학습 방법은 일반적으로 예측이나 분류되는 문제를 해결하며, 문제의 주체인 종속변수에 대해서 레이블이라는 정보를 바탕으로 학습을 진행합니다. 정답값인 레이블을 알고 있으면 저희가 구축한 모델이 정답을 말하는지 오답을 말하는지 알기가 쉬워서 모델의 성능을 평가하기 쉬워집니다.

비지도 학습은 지도학습에서 주어졌던 문제의 주체에 대한 정보를 가진 레이블이 존재하지 않으며 분류나 예측 문제를 해결하기 보다는 비지도 학습 모델이 레이블이 없는 데이터의 특징이나 데이터가 가지는 패턴을 분석하여 나누는 방식의 **군집화(Clustering) 문제**를 해결할 수 있습니다.

또한, 비지도 학습은 고차원의 데이터의 특성(feature)들을 정보의 손실을 최소화하는 방향으로 저차원의 형태로 표현하는 문제인 **차원 축소 문제(Dimensionality reduction)**에도 사용되어집니다.



## 2. 비지도 학습 - 군집화

군집화 문제에 사용되는 알고리즘은 대표적으로 K-means와 DBSCAN 알고리즘이 사용되어 집니다.

### 2.1 K-means

K-means 알고리즘은 K값이 주어졌을 때, 주어진 데이터들 K개의 클러스터로 묶어주는 알고리즘으로 적용되는 방식이 다른 알고리즘에 비해서 단순한 편에 속합니다.

K-means 알고리즘이 동작하는 순서는 아래와 같습니다.

1. 임의의 위치에 K개의 centroid를 좌표에 지정합니다. (초기값 설정)

   ![image](https://user-images.githubusercontent.com/51338268/139422173-74dbe584-3ba6-4eeb-bc9f-f443bfb628e7.png)

2. 데이터들과 K개의 centroid까지의 거리를 모두 구한 뒤, 데이터들은 가까운 거리에 있는 centroid로 클러스터링이 됩니다.                                                 

   - 데이터들의 거리는 유클리드 거리로 구합니다.

   ![image](https://user-images.githubusercontent.com/51338268/139422282-5a322b55-146b-4d98-99bd-9df2b3e8ba4b.png)

3. K개의 클러스터들은 각 클러스터들의 중심점으로 centroid를 이동시킵니다.

   ![image](https://user-images.githubusercontent.com/51338268/139422342-73fb2aca-2c8e-44d5-88ad-3404936e0068.png)

4. 더 이상 centroid가 이동하지 않을때 까지 2,3 과정을 반복합니다.

   ![image](https://user-images.githubusercontent.com/51338268/139422390-f8f44020-24d1-49d8-8749-9f61b674a365.png)

K-means 알고리즘은 위의 과정을 통하여 데이터간의 유사성을 측정하고 비슷한 특성을 지녔다고 판단되는 데이터끼리 클러스터로 묶어주는 작업을 진행하며 이렇게 나온 클러스터는 새로운 파생변수로도 사용이 가능합니다.



### 2.2 K-means 알고리즘의 취약점

K-means 알고리즘은 데이터간의 거리를 기준으로 클러스터를 나누기 때문에 데이터의 분포에 큰 영향을 받는 알고리즘이라서 특수한 분포를 가진 데이터는 제대로 특성을 파악하기 어렵습니다.



<p float="left">
    <img src="https://user-images.githubusercontent.com/51338268/139425456-6a123466-0482-4614-827f-f54b04ddcbe8.png" width="300" />
    <img src="https://user-images.githubusercontent.com/51338268/139425759-209d55f4-ffd9-404e-8e04-fbd5c620ef53.png" width="300" />
</p>
위의 그림은 원 형태의 분포를 가진 특수한 형태의 데이터로 실제 데이터는 왼쪽의 그림과 같이 안과 밖으로 데이터의 특성이 나뉘는게 맞지만, K-means의 경우 오른쪽의 그림과 같이 거리를 기반으로 나누기 때문에 제대로 클러스터링이 이루어지지 않게 됩니다.



<p float="left">
    <img src="https://user-images.githubusercontent.com/51338268/139427879-81f8d3b4-1b66-4bca-8176-e8377883cb4d.png" width="300" />
    <img src="https://user-images.githubusercontent.com/51338268/139427975-4efd7993-1fff-45b0-92ab-9cdf3bd74056.png" width="300" />
</p>

<p float="left">
    <img src="https://user-images.githubusercontent.com/51338268/139428347-4bd19c52-f58b-4b3a-bb14-b10c66c4cba1.png" width="300" />
    <img src="https://user-images.githubusercontent.com/51338268/139428141-3f385094-1cfd-4e5f-b994-6efd329b0bdc.png" width="300" />
</p>

원 형태의 특수한 분포 말고도 초승달 형태의 분포와 대각선 방향의 분포를 가진 분포에서도 K-means 알고리즘은 제 기능을 하지 못하게 됩니다.

이러한 예시를 통하여 K-means 알고리즘이 적합하지 않은 경우를 정리하자면 

- 군집의 개수 K값을 미리 지정해야 하기 때문에 이를 알거나 예측하기 어려운 경우에는 사용하기 어렵다.
- 유클리드 거리가 가까운 데이터끼리 형성되기에 데이터의 분포에 따라 유클리드 거리가 멀면 밀접하게 연관되어 있는 데이터들의 군집화를 수행하기는 어렵다.

지금까지는 군집의 개수를 명시하고 유클리드 거리를 기반으로 군집을 만들었지만, 그렇다면 군집의 개수를 명시하지 않고 밀도 기반으로 군집을 예측하는 방법이 있을까요?



### 2.4 DBSCAN

DBSCAN 알고리즘은 Density Based Spatial Clustering of Applications With Noise의 약자로 가장 널리 알려진 밀도(density) 기반의 군집 알고리즘입니다.

K-means는 클러스터의 개수인 K값을 미리 지정해야 했는데 DBSCAN은 미리 지정할 필요가 없으며 유클리드 거리 기반의 K-means에서 제대로 파악하지 못했던 특수한 형태의 분포도 DBSCAN은 조밀하게 몰려있는 클러스터를 군집화하는 방식으로 문제를 해결할 수 있습니다.

DBSCAN을 이해하기 전에 필요한 변수와 용어를 정리하겠습니다.

- epsilon : 클러스터의 반경(하이퍼파라미터)
- minPts : 클러스터를 이루는 데이터의 최소 개수(하이퍼파라미터)
  - 이 값이 작으면 클러스터가 되기 위해서 요구되는 데이터의 갯수가 줄기 때문에 만들어지는 클러스터가 많아지게 된다.
- core point : epsilon 내에 minPts개 이상의 점이 존재하는 중심점
- border point : 군집의 중심이 되지는 못하지만, 군집에 속하는 점
- noise point : 군집에 포함되지 못하는 점

DBSCAN 알고리즘이 minPts가 4개일 때를 기준으로 순서는 아래와 같습니다.

1. 임의의 점 P를 설정하고, P를 기준으로 epsilon안에 속하는 데이터의 개수를 P를 포함하여  구하여 만일 minPts개 이상의 점이 포함되면 점 P를 core point로 간주하고 원에 포함된 점들을 하나의 클러스터로 묶습니다

   ![image](https://user-images.githubusercontent.com/51338268/139453121-248d9215-6c7b-4c39-b84e-321a6bd6c85a.png)

2. 임의의 점 Q에서 minPts개 미만의 점이 포함되어 있으면 pass 합니다.![image](https://user-images.githubusercontent.com/51338268/139453315-437f438d-275b-4b6e-92ac-30e84c8f01cc.png)

3. 모든 점들에 대해서 1~3번의 과정을 반복하며, 만일 임의의 점 R이 core point가 되고 이점이 기존의 클러스터에 속한다면, 두 개의 클러스터는 연결되어 있다고 간주하여 하나의 클러스터로 묶어줍니다.![image](https://user-images.githubusercontent.com/51338268/139454347-22ed47ef-118a-4f17-9711-f86c4d2a59f0.png)

4. 모든 점에 대해서 클러스터가 끝났는데 어떤 클러스터에도 속하지 못하는 데이터가 존재하면 noise point로 간주하고, 특정 군집에 속하지만 core point가 아닌 점들을 border point라고 부릅니다.![image](https://user-images.githubusercontent.com/51338268/139455952-e5ff2f76-58c6-4f6a-ad3c-9ea2d97d38e8.png)

DBSCAN은 K-means보다 데이터의 특성을 더 잘 파악하여 클러스터링을 시행해주는 알고리즘인데 단점이 있다면 클러스터링을 시킬 데이터의 수가 많아 질수록 수행 시간이 기하급수적으로 늘어나게 됩니다.



## 참고 사이트

- [K-means](https://ko.wikipedia.org/wiki/K-%ED%8F%89%EA%B7%A0_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98)
- [DBSCAN](https://bcho.tistory.com/1205)
- [DBSCAN 시각화](http://primo.ai/index.php?title=Density-Based_Spatial_Clustering_of_Applications_with_Noise_(DBSCAN))
