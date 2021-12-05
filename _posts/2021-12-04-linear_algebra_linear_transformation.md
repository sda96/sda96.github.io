---
title: 선형대수 정리 - 선형변환, 선형시스템
categories: [math]
comments: true
use_math: true
---



## 1. 선형변환(linear transformation)

- 선형변환은 벡터의 크기와 방향을 바꿔주는 함수, 연산(행렬곱)을 의미합니다.

- 수학적 정의는 아래의 두 조건을 만족해야 합니다.

  - $f(x + y) = f(x) + f(y)$​​​​
  - $f(ax) = af(x)$​

- 선형변환의 기하학적 연산은 화소들의 공간적인 관계를 변환시켜주는 것으로 화소들을 재배치하여 영상내의 화소 간의 관계를 변환시켜줍니다.

- 입력 x 와 출력 y가 선형인 경우 Affine 변환이라고 부릅니다

  - Affine 변환은 직선은 그대로 유지되며 평행한 선들은 평행을 유지합니다.
  - 이동, 회전, 확대, 축소, 비틀림등이 모두 Affine 변환에 속합니다.

- 선형변환을 함수라고 생각했을 때, 정의역은 행공간과 영공간을 합친 공간이고 공역은 열공간과 좌영공간이며 치역은 열공간(columns space)입니다.

  - 열공간은 열벡터들의 모든 선형결합의 집합입니다.

  - 열벡터들의 모든 선형결합의 집합은 열공간을 생성(span)합니다.

  - 영공간(null space)는 선형변환을 하여 나온 결과가 0인 벡터들의 집합입니다.

  - 행공간과 영공간은 서로 직교합니다.

    ![image](https://user-images.githubusercontent.com/51338268/144717175-f5f21d8b-bf6d-48f5-a4f1-0ff0a6e229f3.png)

  - 벡터공간의 부분집합이 선형결합에 닫혀있는 경우 부분공간이라고 부릅니다.



## 2. 선형시스템(linear system)

- 선형방정식의 집합(연립 방정식)

- 선형 시스템의 표현 방법

  - 연립 방정식

    $-x_1 + x_2 + x_3 = 0$

    $x_2 - 4x_3 = 4$​

    $-4x_1 + 5x_2 + 8x_3 = -9$

  - 행렬

    $\begin{bmatrix} 1 & -1 & 1 \\ 0 & 1 & -4 \\ -4 & 5 & 8 \end{bmatrix}\begin{bmatrix} x_1 \\ x_2 \\ x_3\end{bmatrix} = \begin{bmatrix} 0 \\ 4 \\ -9 \end{bmatrix}$​

    - 계수 행렬(coefficient matrix)

      $\begin{bmatrix} 1 & -1 & 1 \\ 0 & 1 & -4 \\ -4 & 5 & 8 \end{bmatrix} $

    - 첨가 행렬(augmented matrix)

      $\left[\begin{array}{rrr|r} 1 & -1 & 1 & 0\\ 0 & 1 & -4 & 4 \\ -4 & 5 & 8 & -9 \end{array}\right] $​​

  - 선형결합

    $\begin{bmatrix} 1 \\ 0 \\ -4 \end{bmatrix} x_1 + \begin{bmatrix} -1 \\ 1 \\ 5 \end{bmatrix} x_2 + \begin{bmatrix} 1 \\ -4 \\ 8 \end{bmatrix} x_3= \begin{bmatrix} 0 \\ 4 \\ -9 \end{bmatrix}$​​​​​​

- 선형시스템이 연립 방정식이기 때문에 궁극적인 목적은 연립 방정식의 해를 구하는 것이며 가우스-조르단 소거법을 통하여 해를 구할 수 있습니다.

- 연립 방정식의 해가 나올 수 있는 종류

  - 해가 존재할 때 미지수 벡터 $X = [x_1 ,x_2, x_3]$​의 역행렬이 존재하여 invertible 합니다.
    - 해가 유일할 때는 벡터들이 선형 독립입니다.
    - 해가 무수히 많이 존재할 때 벡터들이 선형 종속입니다.
  - 해가 존재하지 않을 때 미지수 벡터의 역행렬이 존재하지 않기에 non-invertible 하고, 벡터들은 선형 종속이 됩니다.
