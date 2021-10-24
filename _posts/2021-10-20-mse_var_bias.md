---
title: MSE로부터 분산과 편향 도출하기
categories: [math]
comments: true
use_math: true
---



## 1. MSE가 뭐지?

MSE란 Mean Square Error의 줄임말로 해석하면 평균제곱오차입니다. 

여기서 말하는 **오차는 추정량 $\hat{\theta}$​과 대상 모수 $\theta$​ 의 차이**를 말하며 $\hat{\theta}$​ 은 확률변수이고, $\theta$ 는 상수라고 볼 수 있습니다. 

$MSE$는 일반적으로 회귀문제의 비용함수로써 자주 사용되어지며 일반적으로는 $MSE$ 가 작을수록 해당 모델은 모수를 잘 추정하고 있다는 의미입니다.

식으로 표현하면
$$
MSE(\hat{\theta}) = E[(\hat{\theta}-\theta)^2]
$$
추정량 $\hat{\theta}$​​​​​​​​​​ 에 모수 $\theta$​​​​ 를 뺀 뒤 제곱을 한 값들에 기대값을 씌우면 추정량 $\hat\theta$​ 의 모수 $\theta$​ 에 대한 평균제곱오차 $MSE(\hat{\theta})$​​​ ​​​​​가 구해지게 됩니다.



## 2. MSE에서 어떻게 분산과 편향을 도출할까?

식을 하나씩 전개하며 정리해보겠습니다.

1. 우선 기대값 내부의 제곱을 전개합니다.

$$
MSE(\hat{\theta}) = E[\hat{\theta}^2-2\theta\hat{\theta}-\theta^2]
$$

2. 각 원소에 기대값을 적용합니다. 여기서 $\theta$ 는 상수값이기 때문에 기대값 밖으로 빠져나오게 됩니다. 

$$
MSE(\hat{\theta}) = E[\hat{\theta}^2]-2{\theta}E[\hat{\theta}]+\theta^2
$$

3. 분산과 편향으로 정의하기 위해서 추정량의 평균 제곱값을 더하고 빼줍니다.

$$
MSE(\hat{\theta}) = E[\hat{\theta}^2]-2{\theta}E[\hat{\theta}]+\theta^2  +(E(\hat{\theta}))^2 - (E(\hat{\theta}))^2
$$

4. 전개한 값들을 정리합니다.

$$
MSE(\hat{\theta}) = E[\hat{\theta}^2]-(E(\hat{\theta}))^2
+(E(\hat{\theta}) - \theta)^2
$$

5. 정리한 값들의 의미를 살펴보면 추정량의 $MSE(\hat\theta)$ 는 추정량의 분산 $Var(\hat\theta)$ 에 추정량의 편향의 제곱 $(Bias(\hat\theta))^2$​ 을 더한값이 됩니다.

$$
MSE(\hat{\theta}) = Var(\hat\theta) +(Bias(\hat\theta))^2
$$

결과적으로는 추정량의 $MSE(\hat\theta)$​ 는 추정량의 분산과 편향의 제곱을 더한값으로 표현이 가능한 것을 수식으로 정리가 가능하였습니다.

이 과정을 보기 편하게 통째로 이어서 보여드리겠습니다.
$$
\begin{align}
MSE(\hat{\theta}) &= E[(\hat{\theta}-\theta)^2] \\
&=  E[\hat{\theta}^2-2\theta\hat{\theta}-\theta^2] \\
&= E[\hat{\theta}^2]-2{\theta}E[\hat{\theta}]+\theta^2 \\
&= E[\hat{\theta}^2]-2{\theta}E[\hat{\theta}]+\theta^2  +(E(\hat{\theta}))^2 - (E(\hat{\theta}))^2 \\
&= E[\hat{\theta}^2]-(E(\hat{\theta}))^2
+(E(\hat{\theta}) - \theta)^2 \\
&= Var(\hat\theta) +(Bias(\hat\theta))^2
\end{align}
$$


## 3. 최적의 모델을 찾기위한 최소의 MSE

앞서 $MSE$ 는 회귀문제를 해결할 때 비용함수로써 자주 사용되며 $MSE$ 의 값이 낮으면 낮을수록 해당 모델이 모수를 잘 추정하고 있다고 본다고 하였습니다.

$MSE$​​ 가 작아질려면 추정량의 분산이 작을수록 더 좋은 모델이라고 볼 수 있고, 추정량의 편향 또한 작을 수록 심지어는 없으면 더 좋다는 것을 알 수 있습니다.

![image](https://user-images.githubusercontent.com/51338268/138585615-42c33846-3cdb-4028-9fda-7bae1bbd54d3.png)

하지만 분산과 편향은 서로 trade-off 관계이기 때문에 분산이 감소하면 편향이 상승하고 편향이 감소하면 분산이 증가하는 경향을 보입니다.

모델의 복잡도가 낮으면 분산은 작지만 편향은 크게 나타나게 되어 제대로 학습이 이루어지지 않은 모델인 과소적합(Under-fitting)모델이 만들어집니다.

모델의 복잡도가 높으면 분산은 크지만 편향은 작게 나타나는 과대적합(Over-fitting)모델이 만들어지는데 이 말은 훈련 데이터셋에서는 성능이 좋지만 테스트 데이터셋에서는 성능이 좋지 못하여 모델의 일반화 능력이 크게 결여된 모델이 됩니다. 

결론적으로는 분산과 편향은 trade-off 관계이기 때문에 전체오차, 즉 $MSE$ 가 가장 작은 지점을 선택하는 것이 바람직한 모델을 가질 수 있습니다.



## 4. 참고 사이트

- [유튜브 1](https://www.youtube.com/results?search_query=%ED%99%95%EB%A5%A0%EB%B3%80%EC%88%98+%ED%99%95%EB%A5%A0%EB%B6%84%ED%8F%AC)

- [유튜브 2](https://www.youtube.com/watch?v=-CbVagdHqIQ)

- [유튜브 3](https://www.youtube.com/watch?v=mZwszY3kQBg)

- [강의노트](https://cs182sp21.github.io/static/slides/lec-3.pdf)

- [사이트 1](https://medium.com/mighty-data-science-bootcamp/%EC%B5%9C%EC%84%A0%EC%9D%98-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EC%B0%BE%EC%95%84%EC%84%9C-%EB%B6%80%EC%A0%9C-bias%EC%99%80-variance-%EB%AC%B8%EC%A0%9C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-eccbaa9e0f50)

- [사이트 2](https://doublekpark.blogspot.com/2019/01/4-variance-bias-trade-off.html)
