---
title: 까먹기 전에 정리하기 - Pipeline, GridSearchCV
categories: [python]
comments: true
---

일반적으로 데이터가 모델에 입력되어져서 원하는 출력값으로 나오는 과정을 간략하게 일반화하면 아래와 같습니다.

![image](https://user-images.githubusercontent.com/51338268/135743683-b0c6cbe3-4e4e-4e55-ae98-c3c7787ef6be.png)

데이터는 각 단계를 거쳐서 사용자가 원하는 결과가 나오게 되는데 각 과정들은 일부 반복되고 지루한 경우가 발생하며 직접 만들 수는 있지만 귀찮기도 합니다.

이러한 상황에서 Pipeline과 GridSearchCV는 정말 편안한 파이프라인을 만들어주는 고마운 메소드입니다.

## 1. [Pipleline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

Pipeline 메소드는 데이터의 전처리와 모델 구축과정을 간략히 줄어주는 파이프라인을 구축시켜주는 함수입니다. 

![image](https://user-images.githubusercontent.com/51338268/135743731-4cba5003-d617-46fa-8803-3945a3482157.png)

sklearn의 Pipeline 메소드는 ```from sklearn.pipeline import Pipeline  ``` 명령어를 사용해서 불러올 수 있습니다.

``` python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MInMaxScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# 임의의 연습용 분류 데이터를 가져옵니다.
X, y = make_classification(random_state=0)
# 가져온 데이터를 훈련 데이터셋과 테스트 데이터셋으로 분리합니다.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 파이프라인을 설계합니다.
# 입력되는 값은 StandardScaler가 적용되며 적용된 값들은 SVC모델에 들어가 fitting을 시작합니다.
pipe = Pipeline([('scaler', StandardScaler()), 
                 ('scaler', MInMaxScaler())
                 ('svc', SVC())])
pipe.fit(X_train, y_train)

# fitting이 완료된 모델을 가지고 테스트셋으로 점수를 검증합니다.
pipe.score(X_test, y_test)
```

Pipeline이 사용되는 방식은 위와 같으며 위의 예제는 1개의 scaler뿐만 아니라 추가적인 전처리를 원하는 경우 사용되는 방식을 알려주기 위한 예제로 실제로는 1개 정도의 scaler만 사용되어 집니다. 

Pipeline 메소드를 사용하여 데이터의 전처리부터 모델링을 하는 과정을 일련의 파이프라인으로 구축하였습니다.

그렇다면 이제, 동일하게 전처리가 되어진 데이터들이 어떠한 모델에서 가장 좋은 성능을 가지는지 알고 싶어졌습니다.

일반적인 형태로 모델의 성능을 비교하려면 반복되는 구간을 함수로 만들어서 직접 구현하는 방식도 존재합니다. 하지만 직접 구현하려면 시간이 좀 걸리고 귀찮은 점이 많습니다. 

sklearn에서는 이러한 귀찮은 과정을 한번에 적용하여 최적의 모델을 알려주는 GridSearchCV 함수가 존재합니다.

GridsearchCV함수에 Pipeline함수도 적용하여 전처리부터 모델링 과정을 한번에 처리해주는 파이프라인을 구축하는 방법을 알아보겠습니다.

## 2. [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)

![image](https://user-images.githubusercontent.com/51338268/135743814-245d1570-3007-4e37-a655-b0ceb5626c7f.png)

GridSearchCV는 이름 그대로 모델들을 탐욕스럽게 탐색하며 CV(교차검증)을 통해서 사용자가 원하는 최적의 모델을 찾아내는 함수입니다.

이름을 풀어보니 이 함수를 사용하면 사막에서 바늘을 찾게 해주는 마냥 아주 고마운 함수라는 것을 알 수 있습니다.

GrdiSearchCV 함수의 장점을 3가지 정도로 정리하여 말하자면

1. 사용자가 원하는 모델들의 성능을 한번에 비교가 가능합니다.
2. 각 모델들의 파라미터에 변화를 주며 파라미터의 변화에 따른 성능의 변화를 관찰할 수 있습니다.
3. 다양한 종류의 모델과 미세한 조정을 거친 파라미터중에서 사용자가 선택한 척도를 기준으로 최적의 모델을 도출해냅니다.

사용 방식은 아래와 같습니다.

``` python
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

# 데이터를 불러와 훈련 데이터셋과 테스트 데이터셋으로 분리합니다.
X, y = load_digits(return_X_y = True)
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 0)

# 전처리 파이프라인 구축합니다.
pipeline = Pipeline([
                     ('transformer', MinMaxScaler()),
                     ('model', 'passthrough')                   
])    
# 모델들과 각 모델들의 파라미터들을 지정합니다.
params = [
          {"model":(LogisticRegression(),),
           "model__max_iter":[200, 300, 400]},
          {"model":(RandomForestClassifier(),),
           "model__n_estimators":[150, 200, 300]},
          {"model":(SVC(),)}
]
# 원하는 지표를 기준으로 모델 검증 파이프라인 구축합니다.
grid_model = GridSearchCV(pipeline, params, scoring = "accuracy")

# 훈련 데이터셋을 파이프라인에서 정의한 각종 모델에 fitting합니다.
grid_model.fit(x_train, y_train)

# fitting하며 나온 CV의 결과들을 dict타입으로 정리하여 출력합니다.
print(grid_model.cv_results_)

# "accuracy" 지표를 기준으로 가장 좋았던 모델로 테스트셋의 점수를 도출합니다.
grid_model.score(x_test, y_test)
```

GridSearchCV는 사용자가 정의한 파이프라인과 파라미터를 기준으로 가장 성능이 좋았던 모델을 반환하는 방식을 택하고 있으며 각 모델들의 성능의 순서또한 제공하고 있습니다.

다양한 모델과 파라미터들을 손쉽게 적용하고 그 중에서 가장 좋은 성능의 모델을 추천하는 함수인 GridSearchCV의 유일한 단점은 실험하고 싶은 모델과 파라미터의 수가 많아질수록 Grid하게 모델들을 모두 fitting하다보니 시간이 오래걸리게 되는 것 입니다.

그러므로 GridSearchCv는 사용자를 간편하게 만들어주는 도구이지만 적은 수의 데이터로 간단하게 모델의 성능을 비교하거나 파라미터에 따른 모델의 성능 비교가 필요할 때 사용하는 것이 좋다고 생각합니다.

