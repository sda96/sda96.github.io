---
title: 선형대수 정리 - 행렬, 역행렬
categories: [math]
comments: true
use_math: true
---



## 1. 행렬(matrix)

- 행렬 $A$는행(열)벡터들의 결합으로 생각할 수 있습니다.

- **행렬의 종류**

  - 정방 행렬(square matrix) 
    - 행과 열의 개수가 동일한 행렬
    - $A_{n\times n}$

  - 대각 행렬(diagonal matrix)

    - 대각 성분만 존재하고 나머지는 0인 행렬

  - 단위 행렬(identity matrix)

    - 대각 성분이 모두 1이고 나머지가 0인 행렬
    - $A I = IA = A$​

  - 전치 행렬(transpose matrix)

    - 대각 성분을 기준으로 위치가 바뀐 행렬
    - $(A^T)^T = A $​
    - $(A \pm B)^T = A^T \pm B^T $​​​
    - $(kA)^T = kA^T $
    - $(AB)^T = B^TA^T$

  - 대칭 행렬(symmetric matrix)

    - $A = A^T$

  - 직교 행렬(orthogonal matrix)

    - $AA^T = I$

  - 상부삼각행렬(upper triangular matrix)

    $\begin{bmatrix}7 & 1 & -1 \\\ 0 & 2 & 1 \\\ 0 & 0 & 3\end{bmatrix}$​

  - 하부삼각행렬(lower triangular matrix)

    $\begin{bmatrix}7 & 0 & 0 \\\ 1 & 2 & 0 \\\ 2 & 3 & 3\end{bmatrix}$​​​​​​​

  - 행렬의 trace

    - $tr(A) = \sum^N_{k=1} a_{kk}$​
    - $tr(A) = tr(A^T)$
    - $tr(A + B) = tr(A) + tr(B)$​
    - $tr(AB) = tr(BA)$​


- **행렬의 곱셈**
  - 행렬의 곱은 연속 선형변환, 선형변환의 합성이라고 생각할 수 있습니다.
  - 행렬의 곱셈은 앞행렬의 행과 뒷행렬의 열의 내적의 합성되나 결국 앞행렬의 열벡터로 구성되는 선형결합을 의미합니다.
  - $\begin{bmatrix} 1 & 2 \\\ 3 & 4 \end{bmatrix}  \begin{bmatrix} a \\\ c \end{bmatrix} = \begin{bmatrix} 1  \cdot a+ 2 \cdot c  \\\ 3 \cdot a + 4 \cdot c \end{bmatrix} = \begin{bmatrix} 1 \\\ 3 \end{bmatrix}a + \begin{bmatrix} 2 \\\ 4 \end{bmatrix}c   $​​​
- **행렬의 계수**
  - 행렬에서 선형독립인 열(행)벡터의 최대 개수 $n$​을 의미합니다
  - $r(A_{m \times n}) = n$​이면 행렬 $A$​는 비특이(non-singular), 정칙 행렬
  - $r(A_{m \times n}) < n$​​​이면 행렬 $A$​​​는 특이(non-singular), 비정칙 행렬



## 2. 역행렬

- $AA^{-1} = A^{-1}A = I_n$​이 성립하면 행렬 $A$​는 역행렬 $A^{-1}$​이 존재합니다.
- 행렬 $A$의 역행렬이 존재하면 행렬 $A$는 invertible 합니다.
  - 행렬 $A$​가 invertible하면 $A^{-1}, A^{k}, cA, A^{T}$는 모두 invertible 합니다.
  - $(A^{-1})^{-1} = A$​
  - $(A^{-1})^{k} = (A^{k})^{-1}$​​
  - $(cA)^{-1} = \frac{1}{c}A^{-1}$
  - $(A^{-1})^{T} = (A^{T})^{-1}$​​
- 역행렬을 통하여 방정식의 해를 구할 수 있습니다.
  - $B = AX$​​의 수식에서 양변에 계수 벡터인 $A$​​의 역행렬을 행렬곱을 하면 $X = A^{-1}B$이므로 미지수 벡터 $X$를 구할 수 있습니다.
- 역행렬을 구하는 방법
  - $AA^{-1} = I$를 이용합니다.
  - 가우스 - 조르단 소거법을 사용합니다.
    - 가우스 - 조르단 소거법을 사용하여 선형독립인 벡터의 개수를 쉽게 구할 수 있으며 이를 통하여 행렬의 계수를 구할 수 있습니다.
  - 행렬식과 수반 행렬을 활용합니다.
    - 행렬식은 여인자와 소행렬식의 원소들을 곱하고 더한 값입니다.
    - 행렬식은 행렬 $A$가 정방행렬일 아래의 조건이 성립합니다.
      - $det(A^T) = det(A)$​
      - $det(AB) = det(A)det(B)$​
    - 행렬 $A$​가 정칙, 비특이, 정발행렬이면 $det(A^{-1}) = \frac{1}{det(A)}$​ 입니다.
    - 행렬 $A$​​가 비정칙, 특이, 정발행렬이면 $det(A) = 0$​​ 입니다.



## 3. 정리

행렬 $A$가 $n \times n$ 정방행렬 일 때,

- 행렬의 계수 $r(A) < n$으로 비정칙, 특이 행렬이면,

  - 행렬 $A$​의 행렬식은 $det(A) = 0$​​​ 이고, 
  - 행렬식이 0이면 행렬 $A$는 non-invertible 하고,
  - 행렬 $A$의 행(열)벡터들은 선형 종속입니다.

- 행렬의 계수 $r(A) = n$으로 정칙, 비특이 행렬이면,

  - 행렬 $A$의 행렬식은 0이 아니고,

  - 행렬 $A$는 invertibel하기에 역행렬이 존재하고,

  - 행렬 $A$의 행(열)벡터들은 모두 선형 독립이고,

  - 행렬 $A$​의 원소들의 모든 선형결합의 집합 생성할 수 있으며,

  - 생성된 공간의 차원은 $n$​개의 선형독립인 기저 벡터들의 개수이므로 $n$​ 차원입니다.

    
