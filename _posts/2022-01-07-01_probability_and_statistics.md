---
title: 확률과 통계 정리 - 경우의수, 순열, 조합
categories: [math]
comments: true
use_math: true
---



해당 포스팅은 고등학생때 배웠던 확률과 통계의 내용을 복습하기 위해서 만들어졌으며 포스팅의 내용들은 유튜브의 [고등학교 수학 채널 수악중독의 확률과 통계 개념정리](https://www.youtube.com/playlist?list=PLXJ3W1lEGK8Wk4wec4wJA6hFg3-t_div9)내용을 바탕으로 작성되었습니다.



### 1. 경우의 수

- 합의 법칙 : **동시에 일어나지 않는** 두 사건을 더해줍니다.
  - ex) 1~10까지의 카드 2장 선택시 합이 7의 배수인 경우의수
- 곱의 법칙 : **동시에** 혹은 **연속적으로** 일어나는 두 사건을 곱해줍니다.
  - ex) 24의 약수의 갯수 (2와 3이 동시에 발생하는 사건)
  - ex) 주사위 2개를 던질때, 나올 수 있는 경우의 수 (주사위 2개가 연속적으로 발생하는 사건)



### 2. 순열(Permutation)

- $n$개의 데이터 중에서 $r$개를 순서를 고려하여 나열하는 경우의 수를 의미합니다.
  - $_{n}\mathrm{P}_{r} = \frac{n!}{(n-r)!}$​
  - $n! = n \times n-1 \times \cdots \times 1$
  - 순서를 고려한다는 의미는 나열의 $a, b, c$​를 순서를 바꿔서 나열할 수 있는 경우의 수를 구한다고 할 때, $ab$​와 $ba$​는 서로 다른 경우의 수로 본다는 의미입니다.

#### 원순열

- 기존 순열의 경우 시작지점과 끝지점이 명확하지만, 원순열은 누가 시작지점이고 누가 끝지점인지 모르기 때문에 돌아가는 순서가 같으면 똑같은 순서로 봅니다.
- $\frac{n!}{n} = (n-1)!$

#### 중복순열

- $n$개의 데이터중에서 $r$개를 뽑는 경우의 수를 구하는데 중복을 허용하는 경우를 말합니다.
  - $_{n}\mathrm{\Pi}_{r} = n^r$​



### 3. 조합(Combination)

- $n$개의 데이터 중에서 $r$개를 순서롤 고려하지 않고 선택하는 경우의 수를 말합니다.
  - $_{n}\mathrm{C}_{r} = \frac{n!}{(n-r)!r!}$​​
  - 앞선 조합과 다르게 순서를 고려하지 않기 때문에 $ab$​와 $ba$​를 서로 같은 경우의 수로 보게됩니다.

#### 중복조합

- 순서를 고려하지 않고 중복을 허용하는 경우의 수를 구하는 방법입니다.
  - $_{n}\mathrm{H}_{r} = _{n-1 + r}\mathrm{C}_{r}$

#### 이항 정리

- 2개의 항을 n승한 결과를 구하는 방법으로 식으로는 $(a+b)^n$ 입니다. 

  - $(a + b)^n = \sum^{n}_{r=0} {_{n}\mathrm{C}_{r}{a^{{n-r}}b^r}}$​​
  - $_{n}\mathrm{C}_{r}$​을 이항 계수라고 부릅니다.

- 이항정리의 특성

  - $(1+x)^n = _{n}\mathrm{C}_{0}x^0 + _{n}\mathrm{C}_{1}x^1 + \cdots + _{n}\mathrm{C}_{n}x^n$​​​​​​ 일 때, $x$​​​​​가 $1$​​​​인 경우와 $-1$​​​​인 경우를 구해줍니다.

    $2^n = _{n}\mathrm{C}_{0} + _{n}\mathrm{C}_{1} + _{n}\mathrm{C}_{2} +\cdots + _{n}\mathrm{C}_{n}$​

    $0 = _{n}\mathrm{C}_{0} - _{n}\mathrm{C}_{1} + _{n}\mathrm{C}_{2} +\cdots + _{n}\mathrm{C}_{n}$​​

    $T = _{n}\mathrm{C}_{0} + _{n}\mathrm{C}_{2} + _{n}\mathrm{C}_{4} +\cdots + _{n}\mathrm{C}_{n} = _{n}\mathrm{C}_{1} - _{n}\mathrm{C}_{3} + _{n}\mathrm{C}_{5} +\cdots + _{n}\mathrm{C}_{n-1}$​​​​​​

    $2^n = 2T$

    $T = 2^{n-1}$​

    

  - $_{n}\mathrm{C}_{n-r} = _{n}\mathrm{C}_{r}$​​​

    $_{n}\mathrm{C}_{n-r} = \frac{n!}{(n-(n-r))!(n-r)!} = \frac{n!}{(r)!(n-r)!} = \frac{n!}{(n-r)!r!} = _{n}\mathrm{C}_{r}$​

    

  - $_{n-1}\mathrm{C}_{r-1} + _{n-1}\mathrm{C}_{r}= _{n}\mathrm{C}_{r}$ 의 공식은 파스칼의 삼각형을 통하여 이해가 쉽도록 할 수 있습니다.

    [![image](https://user-images.githubusercontent.com/51338268/148480278-deaf5ae3-587e-4ddd-a4e2-e8de50d816ad.png)](https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=vollollov&logNo=220947452823)

    파스칼의 삼각형은 그림과 같이 $(x + 1)^n$​​ 의 전개식의 계수를 정리한 것을 표현한 것으로 위 두 수 10, 10이 아래 가운데 값 20과 같습니다.

    ![image](https://user-images.githubusercontent.com/51338268/148480487-8492dad7-0da3-4c12-b536-feb70e681755.png)

    $_{1}\mathrm{C}_{0} + _{1}\mathrm{C}_{1}= _{2}\mathrm{C}_{1}$​ 이라는 이항정리의 특성을 파악할 수 있습니다.

    ![image](https://user-images.githubusercontent.com/51338268/148480843-0c25899c-a441-4225-a682-5cf82fb4e3da.png)

    계산상의 테크닉으로 삼각형의 하키스틱 공식도 존재하며 식으로 정리하면 $_{n+1}\mathrm{C}_{r+1} = \sum^{n}_{k=r} \ _k\mathrm{C}_{r}$ 이라고 볼 수 있습니다.

    
