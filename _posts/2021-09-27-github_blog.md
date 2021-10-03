---
title: Github 블로그 만들기
categories: [github]
comments: true
---

## 1. 새로운 Repository 만들기

![image](https://user-images.githubusercontent.com/51338268/134861587-f6c41a9f-3702-4e98-beb3-a714e081180f.png)

Github 블로그를 만들기 전에 앞으로 포스팅할 내용들을 넣을 새로운 Public Repository를 만들어야 합니다.

> 단,  repo명을 자신의 user.name.github.io로 만들어야 합니다.

저 같은 경우 지금 만들어진 repo가 존재하기 때문에 빨간색으로 뜨지만 이번에 새로 만들시는 분들은 무리없이 만드실 수 있을 것 입니다.

만들어진 repo를 포스팅 전용 폴더에서 관리하기 쉽도록 하기 위해서 따로 폴더를 만드는 것이 좋습니다.



## 2. 새로 만든 Repository clone 하기

![image](https://user-images.githubusercontent.com/51338268/134868733-e329a0d6-5882-4091-9b27-3cb675990aa9.png)

새로 만든 Repository를 

```
git clone [cloned repository url]
```

 명령어를 사용해서 앞으로 작업하게될 폴더에 복사해놓습니다.

해당 명령어를 사용함으로써 따로 ```git remote```를 사용하지 않고도 해당 폴더의 log는 git 폴더에 기록되게 됩니다.



## 3. 로컬에 Jekyll 설치하기

Jekyll 설치 방법은 운영체제에 따라서 조금의 차이가 있습니다.



### 3.1 운영체제가 Window 인 경우

Window의 경우 Jekyll이 요구하는 패키지 매니저인 GEM이 없기 때문에 따로 설치를 해주어야 합니다.

GEM 패키지 매니저는 Ruby라는 프로그램에서 지원하기 때문에 따로 Ruby를 설치해주어야 합니다.

[Ruby 다운로드 사이트](https://rubyinstaller.org/downloads/)에 들어가서 Ruby + Devkit 이 포함된 최신 버전의 Installer를 설치하면 됩니다.

![image](https://user-images.githubusercontent.com/51338268/134870740-d702848c-0b31-4740-ab19-ac8ce06d97a3.png)

> 단, Add Ruby executables to your PATH를 반드시 체크해주어야 합니다.

Installer로 설치가 완료되면 마지막으로 ```rub dk.rb install```을 할거냐고 물어보는데 [예.] 를 누르면 명령 프롬프트 창이 뜨게 됩니다.

![image](https://user-images.githubusercontent.com/51338268/134871569-82090514-05ef-49a5-af01-05ebe270bd60.png)

Enter를 2번 눌러서 마저 설치를 완료시킵니다.

Ruby의 설치가 완료되면  ```gem install jekyll bundler``` 명령어를 사용하여 로컬에 Jekyll을 설치합니다.



### 3.2 운영체제가 MacOS, Linux, Ubuntu등등... 인 경우

해당 운영체제에는 이미 Ruby가 설치되어 있기에 GEM 패키지 관리자를 바로 사용이 가능합니다.

똑같이 ```gem install jekyll bundler``` 명령어를 사용하여 로컬에 Jekyll을 설치합니다.



## 4. Jekyll 생성하기

1. 포스팅 작업할 폴더 경로에서 ```jekyll new .``` 명령어를 사용하여 Jekyll을 생성합니다.
2. ```bundle install ```
3. ```bundle exec jekyll serve``` 또는 ```jekyll serve```
   - 이 명령어를 사용하면 로컬 서버에서 접속이 가능해집니다.
   - Server address를 구글 url창에 입력하면 접속이 가능하지만 아직 jekyll 테마를 적용하지 않았고 존재하는 파일이 없기에 뜨지 않을 지도 모릅니다.

![image](https://user-images.githubusercontent.com/51338268/134876357-f0ac8c93-9f1b-4ce2-8b6b-faebb2ea269c.png)



## 5. Jekyll 테마 적용

### 5.1 테마 선택

- **[jamstackthemes.dev](https://jamstackthemes.dev/ssg/jekyll/)**

- **[jekyllthemes.org](http://jekyllthemes.org/)**

- **[jekyllthemes.io](https://jekyllthemes.io/)**

- **[jekyll-themes.com](https://jekyll-themes.com/)**

위의 사이트중에서 마음에 드는 테마를 선택하고 선택한 테마의 github로 가서 테마의 repository를 다운로드 받습니다.

### 5.2 테마 적용하기

다운로드 받은 폴더의 내용물들을 모두 선택하여 자신의 git blog repository 폴더에 복사합니다.

마지막으로 테마가 적용되었는지 확인하기 위해서 로컬 서버를 열어서 확인해 주겠습니다.

1. ```bundle install```
2. ```bundle exec jekyll serve``` 또는 ```jekyll serve```
3. url에 http://127.0.0.1:4000/ 검색을 하면 결과를 볼 수 있습니다.



## 6. 참고 사이트

- [Github 블로그 만들기 과정1](https://zeddios.tistory.com/1222)
- [Github 블로그 만들기 과정2](https://zeddios.tistory.com/1223)
- [패키지 매니저 GEM 윈도우에 설치하기](https://blog.psangwoo.com/coding/2017/04/02/install-jekyll-on-windows.html)
- [webrick (LoadError)](https://junho85.pe.kr/1850)
- [Ruby 설치 가이드](https://jujeonghwan.github.io/jekyll/how-to-install-ruby-and-jekyll-on-windows-10-kr/)

