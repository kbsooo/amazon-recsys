## Problem Description
  주어진 dataset은 아마존 사용자와 구매 item 데이터이다. 
rating은 무시하기로 하지만, 고려해도 상관없다.


### 요구사항 1.
Dataset의 특징을 면밀히 파악하고 여러 plan을 생각하라.
적절한 전처리를 수행하라.

### 요구사항2.
GNN 디자인하고 구현하라. 
학습을 위한 적절한 metrics를 선정하라. 
Train 파일들을 이용하여 학습시켜라.


### 요구사항3.
임의의 파일을 받아들여서 다음과 같이 추천 결과를 출력하도록 한다. 

====================
user   item   recommend
1      166    O
4      88     X
 ….
====================
Total recommends = 130/200   
Total not recommends = 70/200


수업 중에 교수가 제공하는 파일을 test하여 나온 결과를 평가한다. 