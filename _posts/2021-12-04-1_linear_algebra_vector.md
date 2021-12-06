---
title: 선형대수 정리 - 스칼라, 벡터, 벡터공간
categories: [math]
comments: true
use_math: true
---



## 1. 스칼라(scalar)

- 스칼라는 하나의 숫자를 의미하며 크기를 나타내는 수학적 도구입니다.



## 2. 벡터(vector)

- 기하학의 벡터 : 크기와 방향을 표현하는 수학적 도구
- 대수적인 벡터 : 크기의 의미보다는 방향을 나타내는 위치인 공간상의 한 점을 의미합니다.
- 공간상의 한점인 벡터가 만들 수 있는 공간은 직선(축)입니다
- 고차원의 기하학적 그림을 대수적(수식적)인 표현으로 변환시켜준 것이 벡터입니다. 

- 벡터는 일반적으로 열벡터 $V = \begin{bmatrix} v_1 \\\ v_2 \\\  \vdots \\\ v_n\end{bmatrix}$​​​​​로 표현이 되며 행벡터는  $V^T = [v_1, v_2, \cdots v_n]$​​​​​​​로 표현되어 집니다.



### 2.1 단위 벡터(unit vector)

- 벡터의 방향을 의미하는 벡터입니다.
- 벡터 $v$가 존재할 때, 벡터 $v$의 크기가 1인 벡터를 단위 벡터라고 부릅니다. 벡터를 단위 벡터로 정규화 하는 방식은 $\frac{v}{\vert \vert v \vert \vert}$입니다.
- 벡터 $v$​의 크기 : $\vert \vert v \vert \vert = \sqrt{v^2_1 + v^2_2 + v^2_3 \cdots v^2_n}$​​​



### 2.2 벡터의 내적(scalar product, dot product)

- 대수적 벡터의 내적은 $A \cdot B = a_1b_1 + a_2b_2 \cdots + a_nb_n$​​ 입니다.
- 기하학적 벡터의 내적은 $A \cdot B = \vert \vert A \vert \vert \vert \vert B \vert \vert cos\theta$​​ 로 표현이 가능하며 $\theta$​는 두 벡터가 이루는 각도를 말합니다.
- 벡터의 내적이 scalar product 라고 부르는 이유는 벡터와 벡터를 내적한 결과가 스칼라이기 때문입니다.
- dot product 라고 부르는 이유는 벡터의 내적 연산 기호가 $\cdot$ 이라서 입니다.
- **벡터의 내적이 가지는 의미**
  - $A \cdot B =0  \Leftrightarrow A \perp B$​ 이 말은 벡터 $A$​와 $B$를 내적한 결과가 $0$이면 두 벡터는 서로 직교이고 또한, 두 벡터 $A, B$의 길이가 1이면 정규 직교라는 의미입니다.
  - 벡터 $A$를 내적하면 새로운 공간(축)에서의 $A$​의 좌표를 구할 수 있습니다.
  - 내적을 함으로써 새로운 공간의 점이 되고 새로운 공간의 점들은 이전의 공간에서 보지 못하였던 새로운 패턴을 형성하게 되어서 새로운 직관을 얻을 수 있게 해줍니다.

- **내적의 활용**
  - 직교 분해
    - 서로 다른 방향의 두 벡터 $A, B$가 존재 할 때, ![image](https://user-images.githubusercontent.com/51338268/144707442-f3dae9b8-5b5c-4cc5-86a5-c8c5fd2d06d9.png)
    - $\vert \vert A \vert \vert cos\theta = \frac{\vert \vert A \vert \vert \vert \vert B \vert \vert cos \theta}{\vert \vert B \vert \vert} = \frac{A \cdot B} {\vert \vert B \vert \vert}$​​​​​ 일 때, 해당 값은 스칼라 이므로 B방향의 벡터로 만들기 위해서 B의 단위 벡터인 $B_{unit} = \frac {B} {\vert \vert B \vert \vert}$​​​를 곱해주면 $A_{사영} = (\frac{A \cdot B}{\vert \vert B \vert \vert ^ 2})B$​​   A사영 벡터를 구할 수 있습니다.
    - A수직은 결국, 벡터 A와 벡터 A사영을 벡터의 뺄셈을 한 결과와 동일 하므로 $A_{수직} = A - A_{사영} = A - (\frac{A \cdot B}{\vert \vert B \vert \vert ^ 2})B$ 입니다.



### 2.3 벡터의 외적

- cross product, vector product 라고 부르며 벡터와 벡터를 외적하게되면 벡터가 나오게 됩니다.
- $A \times B = (\vert \vert A \vert \vert  \vert \vert B \vert \vert sin \theta)E$



## 3. 벡터 공간(vector space)

$V$가 벡터의 합과 스칼라 곱의 연산이 정의되는 공집합이 아닌 벡터들로 이루어진 집합으로 9가지 공리를 만족하면 $V$를 벡터공간이라고 부릅니다. 이때 공리들은 $V$안의 모든 벡터 $u, v, w$와 모든 스칼라 $\alpha, \beta$에 대하여 성립해야 하니다.

1. $u$​와 $v$​의 합인 $u + v$​도 $V$​에 속한다. 	                         (덧셈에 대해 닫혀 있다)
2. 모든 $u, v$에 대하여 $u + v = v + u$  	                        (덧셈에 대한 교환 법칙)
3. $(u + v) + w = u + (v + w)$ 	                                  (덧셈에 대한 결합 법칙)
4. $u + 0 = u$인 영벡터가 $V$에 존재한다                          (영벡터가 존재)
5. $V$상의 모든 $u$에 대하여 $u + (-u) = 0$을 만족하는 $-u$가 존재한다 (덧셈에 대한 역원)
6. $u$에다 스칼라 $\alpha$를 곱한 $\alpha u$도 $V$에 속한다                  (스칼라 곱에 대해 닫혀 있다)
7. $\alpha (u + v) = \alpha u + \alpha v$                                                  (스칼라 곱에 대한 분배법칙)
8. $\alpha(\beta u ) = (\alpha \beta) u$                                                           (스칼라 곱에 대한 결합법칙)
9. $1u = u$                                                                          (1은 스칼라 곱의 항등원)



### 3.1 영벡터공간(zero vector space)

- 하나의 원소 0으로만 이루어진 벡터를 영벡터라고 부릅니다.
- 영벡터공간은 모든 벡터들의 중심입니다.



### 3.2 부분공간(subspace)

- 벡터공간 $V$의 부분집합 $W$​가 다음의 두 연산을 만족할 때,

  - 부분집합 $W$는 영벡터를 포함하고 있고, $0v \in W$
  - 벡터의 합에 대해 닫혀 있고, $u \in W$이고 $v \in W$이면 $u + v \in W$ 
  - 스칼라 곱에 대해 닫혀 있는, $u \in W$이고 $\alpha$가 스칼라 값이면 $\alpha u \in W$
  
  벡터의 합과 스칼라 곱에 대해 닫혀있는 새로운 벡터공간을 이룰 때, $W$를 $V$의 **부분공간**이라고 합니다.

