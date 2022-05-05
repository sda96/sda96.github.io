---
title: Probability Likelihood MLE
categories: [math]
comments: true
use_math: true
---

### 확률분포(Probability Distribution)

“확률분포”는 실수를 정의역으로 하고 확률값(0~1)을 치역으로 일대일대응 관계를 가지는 함수를 말합니다.



### 확률(Probability)

“확률”이란 어떤 고정된 분포 D가 주어졌을 때 주어진 관측값 X가 나타날 가능성을 의미합니다.

확률 = P(관측값 X | 확률분포 D)

확률분포에서는 조건에 해당하는 면적을 확률이라고 부릅니다.

![image](https://user-images.githubusercontent.com/51338268/166993733-cd6ede5f-4e79-4edf-b1b6-db7e668641d3.png)

예를 들어 평균이 32이고 분산이 2.5인 정규분포가 주어졌을 때, 32-34사이로 관측될 확률은 그림의 빨간 영역입니다. 



### 가능도(Likelihood)

“가능도”란 어떤 값이 관측되었을 때, 이것이 어떤 확률 분포에서 왔는지에 대한 확률을 말합니다.

가능도 = L(확률분포 D | 관측값 X)

![image](https://user-images.githubusercontent.com/51338268/166993947-dfd96c6d-92f5-4d20-a80e-281f936a6ece.png)

관측값이 34일 때, 그림의 정규분포에서 해당 관측값이 나왔을 확률은 0.12라고 볼 수 있습니다.

그러므로 위의 그림 확률분포의 **y축은 확률이 아닌 가능도**를 의미한다고 볼 수 있고 **분포의 면적은 확률**을 의미하게 됩니다.



### 최대 우도 추정법(Maximum Likelihood Estimator)

“최대 우도 추정법”은 각 관측값에 대한 총 가능도(모든 가능도의 곱)가 최대가 되게하는 분포를 찾는 모수 추정법입니다.

![image](https://user-images.githubusercontent.com/51338268/166994125-3d020b4b-58fb-4c60-b66f-5a8684df02b1.png)

첫 번째 그림은 좌측 끝에 정규분포를 가지는 확률분포의 경우 각 관측값들의 가능도를 구하고 모두 곱하여 나온 총 자유도를 나타는데 수치가 상당히 작습니다.

![image](https://user-images.githubusercontent.com/51338268/166994244-30d4c390-d726-4ed1-b0e6-f13a2a3cdb2e.png)

두 번째 그림은 확률분포를 옮겨가며 총 자유도를 구한 결과를 나타내며 이중에서 **총 자유도가 가장 크게 나온 확률분포가 모집단의 확률분포와 가장 가깝다고 볼 수 있습니다.**



### 참고자료 

- [확률(probability)과 가능도(likelihood) 그리고 최대우도추정(likelihood maximization)](https://jjangjjong.tistory.com/41)

- [Maximum Likelihood, clearly explained!!!](https://www.youtube.com/watch?v=XepXtl9YKwc&list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9&index=40)
