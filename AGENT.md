## Note
- 대답, 생각 과정, 커밋, 출력물에 대해서 기본적으로 한국어를 사용할 것
- local 환경은 macbook pro m4 16gb ram 모델을 사용하고 있음 (mps 가속 사용 가능함)
- 간단한 실험은 local 에서 진행해도 되지만, 규모가 큰 학습 같은 경우엔 실행 요청을 하면 사용자가 colab 등 환경에서 실행시키고 결과를 알려줄 것임
- macbook pro m4 16gb ram (mps 가속 사용 가능) 에서 실행 가능한 정도면 로컬에서 실행하는걸로
- 사용자의 이해를 돕기 위해 다양한 시각화 방법을 활용할 것
- 의미있는 분기마다 commit 할것 (commit message 자세히!)

## Coding Convention
- 이 프로젝트에서는 uv 가상환경을 사용함 (`$ source .venv/bin/activate` 로 활성화)
- agent 가 테스트하기 위한 script는 python 파일로 작성하여 실행할 것
- **Jupytext를 이용한 코드 동기화**:
    - `.ipynb` 파일은 git에 올리지 않음 (`.gitignore`에 추가됨)
    - 대신 `#%%` 문법을 사용하여 `.py` 파일로 코드를 작성함
    - 작성된 `.py` 파일은 `jupytext`를 통해 즉시 `.ipynb`로 동기화해둠 (`jupytext --sync` 활용)
    - 목적: 사용자가 나중에 해당 노트북을 실행해보거나 결과를 확인할 수 있도록 항상 동기화 상태를 유지함

### #%% 문법 사용법
```python
#%% [markdown]
## this is markdown

#%% [code]
print("this is code")
```

이렇게 하면 colab, kaggle 에서 notebook import 할 때 자동으로 notebook으로 바뀜
또한 jupytext를 통해 로컬에서도 .ipynb와 양방향 동기화가 가능함