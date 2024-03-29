---
title: 최대가능도추정법과 손실함수
categories: [math]
comments: true
use_math: true
---

머신 러닝 모델의 역할은 크기가 유한한 실제 데이터셋의 확률분포에 근사하는 예측 확률 분포를 만드는 것입니다.

머신 러닝 모델의 목표는 이러한 실제 분포에 근사하는 예측 분포를 만들기 위해서 조절 가능한 최적의 파라미터(parameter)의 값을 찾는 것입니다.

최적의 파라미터를 찾는 방법으로는 최대가능도추정법과 최대사후확률추정법이 존재하며 하나씩 알아보도록 하겠습니다.



## 1. 최대 가능도 추정법

최대 가능도 추정법은 Maximal Likelihood Estimation 이라고도 불리며 가능도를 최대로 하는 파라미터를 찾는 방법입니다.

여기서 말하는 가능도를 설명하기 위해서는 필요한 개념을 알아보도록 하겠습니다.

#### 사전 확률 (prior probability)

사전 확률은 데이터를 관찰전 파라미터 공간에 주어진 확률분포 $p(\theta)$입니다.

#### 가능도 (Likelihood)

가능도는 주어진 파라미터 분포 $\theta$​​에 대해서 데이터가 얼마나 근사하는지 나타내는 값으로 수식 표현은 $p(X=x \lvert \theta)$​​​입니다.

가능도는 또한, 파라미터 확률 분포인 $p(\theta)$가 정해졌을 때, $x$라는 데이터가 관찰될 확률을 말하기도 하며 결국에는 확률입니다.

가능도가 높다는 의미는 지정된 파라미터 공간에서 데이터 $x$가 관찰될 확률이 높다는 의미로 이 확률을 최대화 시키는 방향이 곧 예측 확률 분포가 실제 확률 분포에 근사시키는 방향과 동일하게 됩니다.

데이터들의 가능도값을 최대화시키는 방향으로 모델을 학습시키는 방법을 **최대 가능도 추정법(MLE)**, Maximal Likelihood Estimation 이라고 부릅니다.

#### 사후 확률

데이터 집합 $X$​에 주어진 파라미터 $\theta$​의 분포는 $p(\theta \vert X)$​로 표현이 가능하며 데이터를 관찰후 계산되는 확률인 사후 확률이라고 부릅니다.

사전 확률과 가능도를 통하여 사후 확률을 구할 수 있는데, 식으로 표현하면 아래와 같습니다.



$p(\theta \lvert X) = \frac{p(X \lvert \theta)p(\theta)}{p(X)}$​



사후 확률은 사전 확률과 가능도를 곱한 값에 데이터 집합에서 데이터를 뽑을 확률인 $p(X)$를 나눈 값과 동일합니다.

#### 지도학습 모델의 최대 가능도 추정법

지도 학습 모델은 입력값 데이터로부터 예측값과 실제값 사이의 거리가 최소가 되는 파라미터를 구해야 이를 구하기 위해서 최대 가능도 추정법으로 알아보겠습니다.

머신 러닝 모델은 입력데이터로부터 예측값과 실제값 사이에 오차가 발생하는데 이 오차는 관찰된 데이터에 노이즈가 섞여 있기 때문입니다. 

이 노이즈는 $i.i.d N\sim(0,\sigma)$​​​​의 조건을 따르며, 이 조건의 의미는 노이즈끼리는 서로 독립이고 동일한 분포를 가지며 노이즈의 분포는 평균이 $0$​​​이고 분산이 $\sigma^2$​​​​인 정규분포를 따르며 예측값의 분포 또한 노이즈의 분포를 따르기 때문에 평균이 $\theta^Tx_n$이고 분산이 $\sigma$인 정규 분포입니다.

파라미터 $\theta$​와 입력 데이터 $x_n$​이 주어졌을 때, 라벨 $y_n$​을 예측하는 관점에서 보았을 때  n번째의 데이터 포인트의 가능도는 다음과 같습니다.



$p(y_n \lvert \theta, x_n) = N(\theta^Tx_n, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(y_n-\theta^Tx_n)^2}{2\sigma^2})$​



위 식을 통하여 데이터 포인트가 모델 함수로부터 멀어질수록, $y_n - \theta^Tx_n$​의 값이 커질수록 가능도는 기하급수적으로 감소하게 되며 이 말은 $x_n$이 주어졌을 때, $y_n$에 가까운 예측값이 나올 확률이 작아진다는 뜻입니다.

최대 가능도 추정법의 목표는 가능도를 최대화 시키는 파라미터를 찾아야 하기 때문에 , $y_n - \theta^Tx_n$가 최소가 되는 파라미터를 찾아야 합니다.

전체 데이터 포인트의 가능도에 대한 식은 다음과 같습니다.



$p(Y \lvert \theta, X) = \mathrm{\Pi}_n p(y_n \lvert \theta, x_n)$​ 



전체 데이터 포인트의 가능도는 n개의 데이터 포인트들의 가능도는 서로 독립이기 때문에 곱의 연산으로 표현이 가능하며 해당 식은 미분 계산이 어렵기 때문에 $\log$​​를 씌워줌으로써 덧셈연산으로 바꾸어 줍니다.

양변에 $\log$를 씌우는 이유는 로그함수는 단조증가 함수이기 때문에 가능도가 최대일때의 파라미터와 로그 가능도가 최대일때의 파라미터는 동일하기 때문에 $\log$를 씌워도 상관없습니다.
$$
\begin{align}
\log p(Y \lvert \theta, X) &= \sum{\log{p(y_n \lvert \theta, x_n)}} \\
&= \sum \log \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(y_n-\theta^Tx_n)^2}{2\sigma^2}) \\
& = \sum\log \frac{1}{\sqrt{2\pi\sigma^2}} + \sum(-\frac{(y_n-\theta^Tx_n)^2}{2\sigma^2}) \\
&= constant + \frac{1}{2\sigma^2}(-\sum (y_n-\theta^Tx_n)^2)
\end{align}
$$
해당 식에서 조절 가능한 변수인 파라미터는 $\theta$뿐이기 때문에 $\theta$만 있는 조건식으로 바꾸면 아래와 같습니다.
$$
\begin{align}
\log p(Y \lvert \theta, X) &= \arg \max_\theta (-\sum (y_n-\theta^Tx_n)^2) \\
&= \arg \min_\theta(\sum (y_n-\theta^Tx_n)^2)
\end{align}
$$
이 식에 따르면 전체 가능도는 $\sum (y_n-\theta^Tx_n)^2$​가 가장 작을 때의 파라미터를 구하면 예측값의 분포는 실제값의 분포에 근사하다고 볼 수 있습니다.



$MSE = \frac{1}{n}\sum (y_n-\theta^Tx_n)^2$



최대 가능도 추정법에 $\log$​를 씌운 형태는 예측 문제에서의 손실함수인 평균제곱오차(MSE)와 유사한 형태를 보입니다.



## 2. 손실함수란?

손실 함수(Loss function)는 머신 러닝 모델의 성능을 평가하는 지표로써 최대 가능도 추정법의 목표와 비슷하게 예측값과 실제값의 차이가 최소가 되는 파라미터를 찾는 방식으로 머신 러닝이 해결하는 문제에 따라서 사용하는 손실 함수의 종류가 다릅니다.

머신 러닝이 해결하는 문제가 만약 회귀문제인 경우 MSE와 MAE와 같은 손실 함수가 사용되며 분류문제인 경우 엔트로피가 사용되어집니다.

분류 문제에서도 분류하는 범주의 개수에 따라서 사용되는 엔트로피가 달라지는데 0과 1로 분류하는 이진 분류 문제의 경우 이진 교차 엔트로피(binary cross entropy)를 사용하고 다중 분류 문제의 경우에는 교차 엔트로피(cross entropy)를 손실 함수로 사용합니다.



## 3. 교차 엔트로피 비용(Cross Entropy Loss)

다중 분류 문제에서 교차 엔트로피를 손실 함수로 사용하는 이유는 교차 엔트로피가 예측 확률 분포와 실제 확률 분포의 차이를 의미하는 지표로 사용이 가능하기 때문입니다.

교차 엔트로피의 비용을 예시를 들어서 값을 구해보겠습니다.

3개의 클래스를 원-핫 인코딩을 하면 [1,0,0], [0,1,0], [0,0,1]로 바꿀 수 있으며 이 값들은 실제값의 확률 분포라고 할 수 있습니다.

예측값들이 소프트맥스 함수를 적용하여 나온 결과들의 예시를 보면 [0.2, 0.7, 0.1]일 때, 교차 엔트로피를 구하면 다음과 같습니다.


$$
\begin{align}
H(P,Q) &= −∑P(y \lvert X)\log{Q}(X) \\
&= -(0⋅\log{0.2} + 1⋅\log{0.7} + 0⋅\log{0.1}) \\
&= -\log{0.7} \approx 0.357

\end{align}
$$


### Cross Entropy와 Likelihood의 관계

모델의 파라미터를 $\theta$로 놓으면, 모델이 표현하는 확률 분포는 $Q(y \lvert X,θ)$로, 데이터의 실제 분포는 $P(y \lvert X)$로 나타낼 수 있습니다. 그런데 $Q(y \lvert X,θ)$는 데이터셋과 파라미터가 주어졌을 때 예측값의 분포를 나타내므로 모델의 likelihood와 같습니다.
$$
\begin{align}
H(P,Q) &= −∑P(y \lvert X)\log{Q}(y \lvert X,θ) \\
&= ∑P(y \lvert X)(−\log{Q}(y \lvert X,θ))
\end{align}
$$
$X$와 $y$는 데이터셋에 의해 결정되는 값이기 때문에 모델의 식이 바뀌어도 변하지 않습니다. 우리가 바꿀 수 있는 부분은 $-\log Q(\mathbf{y} \lvert X,\theta)$뿐이죠. 그러므로 cross entropy를 최소화하는 파라미터 값을 구하는 것은 결국 negative log likelihood를 최소화하는 파라미터를 구하는 것과 같다고 할 수 있습니다.



