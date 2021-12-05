---
title: 선형대수 정리 - 고유값, 고유 벡터, 고유값 분해
categories: [math]
comments: true
use_math: true
---



## 1. 고유값과 고유 벡터

- $y = Ax$의 식에서 $n \times n $​​​ 정방행렬 $A$​​​를 선형변환으로 봤을 때, 행렬 $A$​​​를 고유 벡터(eigen vector) $v$​​​와 고유값(eigen value) $\lambda$​​​으로 표현되어집니다.

- $Av = \lambda v$​

  $\begin{bmatrix}a_{11}  & \cdots & a_{1n} \\ \vdots & & \vdots \\ a_{n1} & \cdots & a_{nn} \end{bmatrix} \begin{bmatrix} v_1 \\ \vdots \\ v_n \end{bmatrix} = \lambda \begin{bmatrix} v_1 \\ \vdots \\ v_n \end{bmatrix}$​

  열벡터 $v$​에 선형변환 행렬 $A$​를 곱하였을 때, 위 식을 만족하는 상수 $\lambda $​를 고유값, 0이 아닌 열벡터 $v$​를 고유 벡터라고 부릅니다.

- **고유 벡터(eigen vector)**

  - 선형변환을 거쳐서 나온 벡터가 크기는 변하더라도 방향은 바뀌지 않는 벡터를 고유 벡터라고 부릅니다.
  - 고유 벡터는 선형변환이 되어진 공간에서 방향이 바뀌지 않는 중심축의 역할을 하기에 해당 공간을 가장 잘 나타내는 벡터라고 할 수 있습니다.
  - 만약 행렬 $r(A) = n$​​ 인 비특이행렬(정칙행렬)인 경우 고유 벡터들은 $n$ 차원 공간을 생성(span)하는 기저 벡터의 역할을 수행하게 됩니다.

- **고유값(eigen value)**

  - 고유값은 고유 벡터가 변화되는 정도를 의미하며 스칼라입니다.
  - 고유 벡터의 크기에 해당합니다.



### 1.1 특성다항식(characteristic polynomial)

- $Av = \lambda v  $​

  $Av-\lambda v = 0 $​,           $0$은 영벡터

  $ \\ (A-\lambda I)v = 0 $,       $I$​는 단위 행렬

  $\\ det(A-\lambda I)v = 0$​

- $\\ det(A-\lambda I) = 0$​​​​ 은 특성다항식으로 고유치를 구하기 위해서 사용되어지며 특성다항식의 실근은 가우스-조르단 소거법을 통하여 고유값을 계산합니다.

- 계산된 고유값을 통하여 이에 해당하는 고유 벡터를 구할 수 있습니다.

- 나올 수 있는 고유값의 개수는 고유 공간의 차원의 수가 됩니다.

- 행렬 $A$​​가 삼각행렬인 경우 고유값은 주 대각선 성분들의 곱으로 표현되어집니다.

  - $Au = \lambda_i u$

    $\prod^n_{i=1} \lambda_i(A) = det(A)$​n

    $\sum^n_{i=1} \lambda_i(A) = tr(A)$​



## 2. 고유값 분해(eigen decomposition)

- $Av_i = \lambda_i v_i ,\quad i = 1, 2, 3, 4, \cdots n$

  $A[v_1, v_2 \cdots v_n] = [\lambda_1v_1, \lambda_2v_2 \cdots \lambda_nv_n] $​​

  $A[v_1, v_2 \cdots v_n] = [v_1, v_2 , \cdots, v_n]\begin{bmatrix}\lambda_1 & 0 & \cdots & 0 \\ 0 & \lambda_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda_n \end{bmatrix}$​​

  정방행렬 $A$​​의 고유 벡터들을 열벡터로, 고유 벡터들의 집합을 나타내는 행렬 $P$​​​, 고유값을 대각 원소로 가지는 대각 행렬 $\and$​​​ 라고 할 때, 다음과 같은 수식으로 표현이 가능합니다.

  $AP = P\and$​

  $A = P\and P^{-1}$​

  이와 같이 행렬 $A$​​를 자신의 고유 벡터들을 열벡터로 하는 행렬과 고유값을 대각원소로 하는 행렬의 곱으로 대각화 분해가 가능한데 이를 **고유값 분해**라고 부릅니다. 

- 고유값 분해가 가능한 조건은 행렬 $A$​가 $n \times n$​ 정방행렬이면서, $n$​개의 선형독립인 고유 벡터를 가지고 있어야 합니다.

  -  $n \times n$ 정방행렬에서 선형 독립인 벡터의 개수가 $n$개라는 말은 행렬 $A$의 계수가 $n$이어야 한다는 의미로 $r(A) = n$이면 행렬 $A$​​는 비특이, 정칙행렬이어야 합니다.

- $n \times n$인 두 행렬 $A$ 와 $\and$는 행렬 $P$​가 비특이, 정칙행렬이면 행렬 $A$와 행렬 $\and$는 닮음행렬입니다.

  - 행렬 $A$와 $\and$가 닮음행렬이면 서로 동일한 고유값을 지니며 각 고유값의 대수적 중복도 또한 같습니다.
  - $n \times n$인 정방행렬 $A$가 임의의 대각 행렬과 닮음 행렬이면 행렬 $A$​​는 대각화 가능(diagonalizable) 합니다.
    - 행렬을 대각화하게 되면 행렬식, 행렬의 거듭제곱, 역행렬, 행렬의 다항식을 계산하기 수월해집니다.
  - $n \times n$인 정방행렬 $A$가 대각화 가능하기 위한 필요충분조건은 행렬 $A$가 $n$​개의 선형독립인 고유 벡터의 집합을 가져야 합니다.

- **대칭행렬과 고유값 분해**

  - $n \times n$​​ 정방행렬 $A$​​중에서 $A = A^T$​​가 성립되는 대칭행렬이면 서로 다른 고유값에 대응하는 고유 벡터들은 서로 직교(orthogonal)하게 됩니다.

  - 직교행렬을 이용한 고유값 대각화가 가능해집니다.

    $A = P \and P^{-1}$​ ​​

    $A = P\and P^T$ , 단, $PP^T = E $​ 인 경우

  - 행렬 $A$​​가 정방행렬이면서 대칭행렬이면 모두 고유값 분해가 가능합니다.

  - Gram-Schmidt (직교 대각화) 과정은 내적공간에서 유한개의 벡터 집합을 직교정규기저로 변환하는 방법입니다.



## 정리

- 행렬 $A$가 $n \times n$ 정방행렬일 때, 행렬의 계수 $r(A)$가 $n$​ 보다 작으면 행렬 $A$는 비정칙, 특이(singular)행렬이고,
- 행렬 $A$의 행렬식은 0이라서 역행렬이 존재하지 않는 비가역적(non-invertible)하고 고유벡터의 개수가 $n$개 보다 적기 때문에 대각화도 가능하지 않습니다.
- 행렬 $A$의 행(열)벡터들은 서로 선형 종속의 관계를 가지고 있습니다.
- 행렬 $A$를 기약행 사다리꼴로 변형하면 하나 이상의 행과 열이 0으로 채워져있습니다.



## 참고 사이트

- https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-19-%ED%96%89%EB%A0%AC
- https://darkpgmr.tistory.com/105
- https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix





