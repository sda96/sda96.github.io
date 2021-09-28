---
title: Python을 사용하는 이유
categories: [python]
comments: true
---



파이썬을 사용하는 이유는 일반적으로 쉽다! 다양한 패키지가 존재하여 직접 코드 생산성이 높다! 등의 장점이 있습니다.

하지만, 이러한 흔한 이유 말고도 파이썬의 장점은 더 많을 거라고 생각합니다!

파이썬 공식 문서에서 언급하는 파이썬의 장점은 무엇인지 알아보겠습니다!



출처 : [파이썬 공식 튜토리얼 문서](https://docs.python.org/3/tutorial/index.html)

> Python is an easy to learn, powerful programming language. It has efficient high-level data structures and a simple but effective approach to object-oriented programming. Python’s elegant syntax and dynamic typing, together with its interpreted nature, make it an ideal language for scripting and rapid application development in many areas on most platforms

파이썬은 배우기 <span style="color:blue">쉬운 빠워뿔한 프로그래밍 언어</span>입니다.

파이썬은 <span style="color:red">효율적인 고수준의 자료 구조</span>와 간단하지만 효과적인 접근방법인 <span style="color:red">객체지향 프로그래밍</span>이라는 특징을 가졌습니다.

파이썬의 <span style="color:red">우아한 구문과 동적 타이핑</span>으로 해석되어진 자연스러움은 파이썬을 스크립팅과 <span style="color:red">대부분의 플랫폼에서 빠른 애플리케이션의 개발</span>이 가능해지도록 만들어 졌습니다.

> The Python interpreter and the extensive standard library are freely available in source or binary form for all major platforms from the Python Web site, https://www.python.org/, and may be freely distributed. The same site also contains distributions of and pointers to many free third party Python modules, programs and tools, and additional documentation

파이썬 인터프리터와 확장성 좋은 기존 라이브러리들은 공식사이트에서 <span style="color:blue">주요 플랫폼들에서 사용 가능한 소스 코드 또는 바이너리 폼들을 무료로 이용 가능하며 배포 또한 가능</span>합니다.

동일한 사이트들 또한 배포와 많은 서드파티 파이썬 모듈, 프로그램, 툴, 추가적인 문서 또한 포함 하고 있습니다.

> The Python interpreter is easily extended with new functions and data types implemented in C or C++ (or other languages callable from C). Python is also suitable as an extension language for customizable applications

파이썬 인터프리터는 새로운 함수로 쉽게 확장되어지고 데이터 타입들은 C 또는 C++안에서 실행되어집니다.

파이썬은 또한 직접 제작 가능한 애플리케이션 제작을 위한 <span style="color:red">확장성 있는 언어</span>로써 적절합니다.

출처 : [파이썬 식욕 돋구기](https://docs.python.org/3/tutorial/appetite.html)

> Python is an interpreted language, which can save you considerable time during program development because no compilation and linking is necessary. The interpreter can be used interactively, which makes it easy to experiment with features of the language, to write throw-away programs, or to test functions during bottom-up program development

파이썬은 <span style="color:red">인터프리터 언어</span>입니다. 인터프리터 언어는 프로그램 개발 시간 동안 컴파일이 없고, 링킹은 필수적이기 떄문에 당신이 고민할 시간을 아낄 수 있습니다.

인터프리터는 인터렉티브하게 사용가능하고 언어의 특성 실험하거나, 프로그램을 작성하거나, 바텀업 방식의 프로그램 개발에서 함수의 시험등을 쉽게 만들어 줍니다.

앞서 언급한 장점과 파이썬 공식 문서에서도 겹치는 내용은 <span style="color:blue">파랑색</span>으로 표시하였고 새롭게 보이는 장점들은  <span style="color:red">빨강색</span>으로 표시했습니다.

파이썬은 배우기 쉽다고 나와 있으며 주요 플랫폼들에서도 사용 가능한 소스코드와 바이너리 폼들이 많이 배포되고 있기에 다양한 패키지들이 쏟아져 나오고 있다고 합니다.

하지만, 그것 말고도 새롭게 보이는 장점으로는 4가지가 있었습니다.

1. 효율적인 고수준의 자료 구조
2. 객체 지향 프로그래밍
3. 우아한 구문과 동적 타이핑
4. 다양한 플랫폼에서도 개발 가능한 확장성 있는 언어
5. 인터프리터 언어

공식문서를 살펴보며 새로운 4가지의 장점을 발견할 수 있었습니다. 그렇다면 각 장점들이 의미하는  바가 무엇인지 용어를 좀 더 깊게 파보겠습니다.

## 1. **효율적인 고수준의 자료 구조**

자료구조(data structures)란 자료를 보관하고 정리하기 위한 코드 구조로 자료에 접근하거나, 변형하기 더 쉽게 해줍니다.

자료구조는 어떻게 자료를 수집할지, 어떻게 자료간의 관계를 설명하고 유기적으로 활용할지 고민을 합니다.

문제를 해결하는데 있어서 처해진 상황에 맞게 적절한 자료구조를 선택할 능력이 있어야 합니다.

파이썬은 다양한 자료형을 지원하는데 그 중에서 자료구조를 표현하기 쉬운 컨테이너 자료형이 존재합니다.

- 컨테이너 자료형

1. List - 배열과 비슷한 구조로 mutable한 특성을 지님
2. Tuple - immutable한 List
3. Set - 순서와 중복이 없으며 집합 연산이 가능한 배열
4. Dictionary - key 와 value 가 쌍을 이루는 형태로 해시 테이블과 비슷한 형태 입니다.

- 번외 - collections library

[참고 사이트 1](https://www.educative.io/blog/8-python-data-structures)

[참고 사이트 2](https://www.edureka.co/blog/data-structures-in-python/)

[참고 사이트 3](https://realpython.com/python-data-structures/#conclusion-python-data-structures)

## 2. 객체 지향 프로그래밍

[파이썬의 공식 문서](https://docs.python.org/3/reference/datamodel.html)에서 말하는 객체는 아래와 같습니다.

객체는 데이터에 대한 파이썬의 추상화입니다.

파이썬 프로그램의 모든 데이터는 객체로 표현되어지거나 객체간의 관계로 표현되어집니다.

모든 객체는 identity(ID), type, value을 가지고 있습니다.

- identity
  - 생성되면 변하지 않는 것으로 객체의 메모리 주소라고 생각하면 됩니다
- type
  - type은 자료형으로 객체의 type은 객체의 연산을 결정하고, 또한 이용 가능한 value 들을 정의합니다.
- value
  - 객체의 value는 변화가 가능합니다. 변화 가능한 값을 가진 객체들을 mutable (가변성) 하다고 부르고 그렇지 않은 경우 immutable 하다고 부릅니다.
  - mutable 인지 immutable 인지는 객체의 type에 의해서 결정됩니다.

파이썬의 공식 문서의 경우 객체는 메모리 주소, 자료형 그리고 값으로 이루어져 있다고 합니다.

파이썬이 객체 지향 언어라는 의미는 파이썬 내부의 모든 프로그램은 객체로 이루어져 있으며 만들고 싶은 프로그램들 또한 새로운 자료형을 정의하여 새로운 객체를 만들 수 있다는 의미입니다.

stackoverflow 에도 객체에 대한 내용이 나오는데 자료형의 중요성을 알 수 있는 내용이 있습니다.

출처 - [stackoverflow : What is Object?](https://stackoverflow.com/questions/56310092/what-is-an-object-in-python)

출처 - [위키백과 : 객체 지향 프로그래밍](https://ko.wikipedia.org/wiki/객체_지향_프로그래밍#자료_추상화)

> 객체 지향 프로그래밍(영어: Object-Oriented Programming, OOP)은 컴퓨터 프로그래밍의 패러다임 중 하나이다. 객체 지향 프로그래밍은 컴퓨터 프로그램을 명령어의 목록으로 보는 시각에서 벗어나 여러 개의 독립된 단위, 즉 "객체"들의 모임으로 파악하고자 하는 것이다. 각각의 객체는 메시지를 주고받고, 데이터를 처리할 수 있다.

- 기본 구성 요소
  - 클래스 : 객체 지향 프로그램의 기본적인 사용자 정의 자료형이라고 할 수 있으며 속성(attribute)과 행위(파이썬은 method)로 이루어져 있습니다.
  - 인스턴스 : 실제 메모리에 할당되는 객체로 클래스에서 정의한 자료형은 가진 객체입니다.
- 특징
  - 자료 추상화 : 불필요한 정보는 숨기고 중요한 정보는 표현하여 프로그램을 간단히 만들게 해줍니다.
  - 상속 : 새로운 클래스가 기존의 클래스의 자료와 연산을 이용 할 수 있게 해주는 기능으로 다중 상속도 가능합니다.
  - 다형성 개념 : 어떤 한 요소에 여러 개념을 넣어 놓은 것으로 일반적으로 오버라이딩, 오버로딩을 의미합니다.
  - 동적 바인딩 : 실행 시간 중에 일어나거나 실행 과정에서 변경될 수 있는 바인딩으로 컴파일 시간에 완료되어 변화하지 않는 정적 바인딩과 대비되는 개념입니다.
- 장점
  - 소프트웨어 공학의 관점에서 S/W의 질을 향상하기 위해 강한 응집력과 약한 결합력을 지향합니다
  - OOP의 경우 하나의 문제 해결을 위한 데이터를 클래스에 모아 놓은 자료형을 사용함으로써 응집력을 강화하고 클래스간 독립적인 디자인을 함으로써 결합력을 약하게 합니다.

## 3. 우아한 구문과 동적 타이핑

우아한 구문의 출처 - [위키 백과](https://en.wikipedia.org/wiki/Python_syntax_and_semantics), [Data-Flair](https://data-flair.training/blogs/python-syntax-semantics/)

파이썬이라는 언어의 디자인 철학, 특성등이 설명되어 있는 링크로 시간의 여유가 있을 때 보면 좋을거라 생각됩니다.

동적 타이핑의 출처 - [미디엄 블로그](https://towardsdatascience.com/dynamic-typing-in-python-307f7c22b24e)

정적 타이핑 언어의 변수들의 타입들은 컴파일이 되는 단계에서 결정되어 집니다.

대부분의 언어들은 정적 타이핑 모델을 지원하여 프로그래머들은 각 변수들의 자료형을 명확히 해야 합니다.

반면에, 파이썬의 자료형들은 컴파일 단계와 반대되는 런타임 단계에서 결정되어 지고, 프로그래머들은 변수를 명확히 할 필요가 없습니다.

## 4. 다양한 플랫폼에서도 개발 가능한 확장성 있는 언어

출처 : [파이썬 식욕 돋구기](https://docs.python.org/3/tutorial/appetite.html)

> Python is simpler to use, available on Windows, Mac OS X, and Unix operating systems, and will help you get the job done more quickly.

여기서 말하는 플랫폼이란 Windows, Mac OS X, Unix 와 같은 OS 운영체제를 의미하며 운영체제의 구분 없이 어디서든지 파이썬을 사용할 수 있다는 의미입니다.

## 5. 인터프리터 언어

사람이 알 수 있는 고급언어(Source Code)를 기계만 알 수 있는 저급언어(기계어)로 변환시켜주는 방법으로 컴파일러와 인터프리터가 있습니다.

- 컴파일러
  - Source Code 전체를 한번에 하드웨어가 처리하기 용이한 형태인 기계어로 바꾼 뒤 실행하는 방법
  - 원시코드의 크기가 너무 크면 실행 속도가 느릴 수 있음
  - 플랫폼에 따라서 다르게 작성해야 해서 이식성이 떨어짐
- 인터프리터
  - 인터프리터는 Source Code를 한번에 한 줄씩 읽어 들여서 중간 코드로 번역한 뒤 바로 실행하는 방법
  - 컴파일러 방식과 다르게 필요한 부분의 고급 프로그램만 선택하여 실행 가능하여 조건부 실행 속도가 빠를 수 있음
  - 인터프리터는 대화형 프로그래밍이 가능하기에 교육용으로도 많이 쓰임
