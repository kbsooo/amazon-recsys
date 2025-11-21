# Amazon GNN 추천 시스템 프로젝트

## 📋 프로젝트 개요

Amazon 사용자-아이템 구매 데이터를 활용한 Graph Neural Network (GNN) 기반 추천 시스템입니다.

### 핵심 특징

- **사용자별 맞춤 추천 개수**: 기존 interaction의 50% 이하로 제한
- **Cold-start 처리**: Interaction ≤ 10인 사용자는 무조건 2개 추천
- **다양한 GNN 모델 비교**: LightGCN, NGCF, SimpleGCN
- **풍부한 평가 지표**: Recall, NDCG, Precision, MRR, Hit Rate, Coverage
- **시각화**: 학습 곡선, 모델 비교, 사용자 그룹별 성능, t-SNE embedding 등

### 데이터셋

- **총 Interactions**: 568,263
- **사용자 수**: 256,009
- **아이템 수**: 74,233
- **Sparsity**: 99.997%

## 🗂 프로젝트 구조

```
c4-5/
├── README.md                      # 이 파일
├── model_research.md              # GNN 모델 7종 비교 분석
│
├── data_analysis.ipynb            # 탐색적 데이터 분석 (EDA)
│
├── preprocessing.py               # 데이터 전처리 모듈
│   ├── DataPreprocessor class
│   ├── User/Item 인코딩
│   ├── Train/Val/Test 분할
│   └── K값(추천 개수) 계산
│
├── models.py                      # GNN 모델 구현
│   ├── LightGCN (주력 모델)
│   ├── NGCF (비교 baseline)
│   └── SimpleGCN (기본 baseline)
│
├── train.py                       # 학습 파이프라인
│   ├── BPR Loss
│   ├── Negative Sampling
│   ├── Early Stopping
│   └── 학습 곡선 시각화
│
├── evaluate.py                    # 평가 시스템
│   ├── Recall@K
│   ├── NDCG@K
│   ├── Precision@K
│   ├── MRR (Mean Reciprocal Rank)
│   ├── Hit Rate@K
│   ├── Coverage
│   └── 사용자 그룹별 평가
│
├── recommend.py                   # 추천 생성 시스템
│   ├── 추천 생성
│   ├── 평가 규칙 준수
│   └── 결과 포맷팅
│
├── visualize.py                   # 시각화 유틸리티
│   ├── 데이터 개요
│   ├── K값 분포
│   ├── 모델 비교 차트
│   ├── 추천 결과 분석
│   └── t-SNE embedding
│
└── amazon_recsys_final.ipynb      # 최종 통합 노트북 (예정)
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# uv 가상환경 활성화
source .venv/bin/activate

# 필수 패키지 설치
pip install numpy pandas matplotlib seaborn torch torch-geometric scikit-learn tqdm
```

### 2. 데이터 전처리

```python
from preprocessing import load_and_preprocess_data

# 데이터 로딩 및 전처리
preprocessor, train_df, val_df, test_df, k_values = load_and_preprocess_data(
    data_path='../data/amazon_train.csv',
    good_rating_threshold=4.0,
    cold_start_threshold=10,
    recommend_ratio=0.5,
    min_recommend=2
)
```

### 3. 그래프 구축

```python
import torch
from preprocessing import create_edge_set

# Train edges
train_edges = create_edge_set(train_df)
edge_list = list(train_edges)

# Bipartite graph edges
users = [e[0] for e in edge_list]
items = [e[1] + preprocessor.n_users for e in edge_list]  # offset

# Edge index (bidirectional)
edge_index = torch.tensor([
    users + items,
    items + users
], dtype=torch.long)
```

### 4. 모델 학습

```python
from models import LightGCN
from train import GNNTrainer

# 모델 초기화
model = LightGCN(
    n_users=preprocessor.n_users,
    n_items=preprocessor.n_items,
    embedding_dim=64,
    n_layers=3,
    device='mps'  # or 'cuda' or 'cpu'
)

# 학습
trainer = GNNTrainer(model, edge_index, device='mps')
trainer.fit(
    train_df, val_df, 
    n_items=preprocessor.n_items,
    epochs=50,
    batch_size=1024,
    learning_rate=0.001
)

# 학습 곡선 시각화
trainer.plot_training_history()
```

### 5. 모델 평가

```python
from evaluate import GNNEvaluator

# 평가
evaluator = GNNEvaluator(model, edge_index, device='mps')
metrics = evaluator.evaluate(train_df, test_df, k=20)

print(f"Recall@20: {metrics['recall@20']:.4f}")
print(f"NDCG@20: {metrics['ndcg@20']:.4f}")
print(f"Precision@20: {metrics['precision@20']:.4f}")

# 사용자 그룹별 평가
group_metrics = evaluator.evaluate_by_user_group(train_df, test_df, k=20)
```

### 6. 추천 생성

```python
from recommend import RecommendationSystem

# 추천 시스템 초기화
rec_sys = RecommendationSystem(model, preprocessor, edge_index, device='mps')

# 추천 생성
result_df = rec_sys.generate_recommendations_for_test(train_df, test_df, k_values)

# 출력 형식
print(rec_sys.format_output(result_df))

# CSV 저장
rec_sys.save_recommendations(result_df, 'recommendations.csv')
```

### 7. 시각화

```python
from visualize import VisualizationUtils

viz = VisualizationUtils()

# 데이터 개요
viz.plot_data_overview(train_df, val_df, test_df)

# K값 분포
viz.plot_k_distribution(k_values)

# 추천 결과 분석
viz.plot_recommendation_analysis(result_df)

# Embedding 시각화 (t-SNE)
viz.plot_embeddings_tsne(model, edge_index, n_samples=1000)
```

## 📊 주요 모델 비교

### LightGCN (주력 모델)
- **특징**: 순수 neighborhood aggregation, 가장 단순하고 효율적
- **장점**: Sparse 데이터에 강함, 빠른 학습, SOTA 성능
- **적합성**: ⭐⭐⭐⭐⭐

### NGCF (Neural Graph Collaborative Filtering)
- **특징**: Feature transformation + Non-linear activation
- **장점**: High-order connectivity 학습
- **단점**: 복잡도 높음, LightGCN보다 성능 낮음
- **적합성**: ⭐⭐⭐

### SimpleGCN (Baseline)
- **특징**: 기본적인 GCN 구조
- **장점**: 구현 단순, 이해 용이
- **적합성**: ⭐⭐

## 📈 평가 지표

- **Recall@K**: 실제 관심 아이템 중 추천된 비율
- **NDCG@K**: 순위를 고려한 평가 (Normalized Discounted Cumulative Gain)
- **Precision@K**: 추천 중 실제 관심 아이템 비율
- **MRR**: 첫 번째 관련 아이템의 순위 역수
- **Hit Rate@K**: 추천에 관련 아이템이 하나라도 있는지
- **Coverage**: 전체 아이템 중 추천에 포함된 비율

## 🔬 실험 계획

1. **Baseline 구축**: SimpleGCN으로 기본 성능 확인
2. **주력 모델 학습**: LightGCN 학습 및 최적화
3. **비교 실험**: NGCF와 성능 비교
4. **하이퍼파라미터 튜닝**: Embedding dimension, layer 수, learning rate 등
5. **사용자 그룹별 분석**: Cold-start vs Regular vs Active users
6. **최종 모델 선정**: 종합 평가를 통한 최적 모델 선택

## 📝 평가 규칙

### 추천 개수 계산
```python
def get_k_for_user(interaction_count):
    if interaction_count <= 10:
        return 2  # Cold-start
    else:
        return int(interaction_count * 0.5)  # 50% 제한
```

### 출력 형식
```
==================================================
user            item                 recommend    
==================================================
A395BORC6F...   B000UA0QIQ           O
A1UQRSCLF8...   B006K2ZZ7K           X
...
==================================================
Total recommends = 130/200
Total not recommends = 70/200
==================================================
```

## 💡 주요 인사이트

1. **데이터 특성**
   - 매우 희소한 그래프 (Sparsity 99.997%)
   - Long-tail 분포 (소수 인기 아이템 집중)
   - 약 40% 사용자가 Cold-start

2. **모델 선정 이유**
   - LightGCN: Sparse 데이터에 최적화, 효율성, SOTA 성능
   - 단순한 구조로 과적합 방지
   - 메모리 효율적 (16GB RAM 환경)

3. **평가 전략**
   - 다양한 지표로 종합 평가
   - 사용자 그룹별 성능 분석
   - 실제 추천 시스템 운영 시나리오 반영

## 🛠 기술 스택

- **Python 3.11+**
- **PyTorch**: 딥러닝 프레임워크
- **PyTorch Geometric**: GNN 구현
- **Pandas/NumPy**: 데이터 처리
- **Matplotlib/Seaborn**: 시각화
- **scikit-learn**: t-SNE 등

## 👥 참고 자료

- **LightGCN**: He et al., "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation" (SIGIR 2020)
- **NGCF**: Wang et al., "Neural Graph Collaborative Filtering" (SIGIR 2019)
- **BPR Loss**: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (UAI 2009)

## 📄 라이선스

이 프로젝트는 교육 목적으로 작성되었습니다.

## ✉️ 문의

프로젝트 관련 문의사항이 있으시면 이슈를 등록해주세요.

---

**Last Updated**: 2025-11-21
