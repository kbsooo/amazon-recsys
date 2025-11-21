# GNN 모델 후보 조사 및 비교

## 추천 시스템용 GNN 모델 후보군

### 1. LightGCN (Light Graph Convolutional Network)
**논문**: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation (SIGIR 2020)

**특징**:
- NGCF를 단순화한 모델
- Non-linear activation과 feature transformation 제거
- 순수 neighborhood aggregation에만 집중
- Layer-wise embedding을 평균하여 최종 embedding 생성

**장점**:
- 🔥 **매우 Simple & Efficient**: 구현이 쉽고 학습 속도가 빠름
- 🔥 **Sparse 데이터에 강함**: 희소한 그래프에서도 우수한 성능
- 🔥 **State-of-the-art 성능**: 많은 벤치마크에서 최고 성능
- 🔥 **메모리 효율적**: 복잡한 연산이 없어 메모리 사용량 적음
- Oversmoothing/Overfitting 문제 완화

**단점**:
- 단순한 구조로 복잡한 패턴 학습에 한계
- Side information(attributes) 활용 어려움

**적합성 (우리 프로젝트)**:
✅ 256K 사용자 × 74K 아이템의 **극도로 희소한 그래프**
✅ MacBook M4 16GB에서 **학습 가능한 효율성**
✅ Collaborative filtering 중심 (rating만 사용)

---

### 2. NGCF (Neural Graph Collaborative Filtering)
**논문**: Neural Graph Collaborative Filtering (SIGIR 2019)

**특징**:
- User-item bipartite graph에 GCN 적용
- High-order connectivity 명시적 모델링
- Multiple layers로 multi-hop relationships 학습

**장점**:
- 고차원 관계 학습 가능
- 이론적으로 탄탄한 기반

**단점**:
- ⚠️ **복잡도 높음**: LightGCN보다 느리고 무거움
- LightGCN에 성능이 밀림 (실험적으로 검증됨)
- Overfitting/Oversmoothing 문제

**적합성 (우리 프로젝트)**:
⚠️ LightGCN의 등장으로 대체됨
❌ 복잡도 대비 성능 향상 미미

---

### 3. GraphSAGE
**논문**: Inductive Representation Learning on Large Graphs (NIPS 2017)

**특징**:
- Inductive learning framework
- Sampling + Aggregation 전략
- 새로운 노드에 대한 generalization 가능

**장점**:
- 🔥 **Inductive capability**: 새로운 사용자/아이템 처리 가능
- 🔥 **Scalable**: 대규모 그래프에 적합
- 다양한 aggregator (mean, LSTM, pooling)

**단점**:
- Collaborative filtering에 최적화되지 않음
- LightGCN보다 복잡

**적합성 (우리 프로젝트)**:
✅ Cold-start 문제 해결에 유용
⚠️ 하지만 우리는 test 시 새로운 노드 없음

---

### 4. GAT (Graph Attention Networks)
**논문**: Graph Attention Networks (ICLR 2018)

**특징**:
- Attention mechanism으로 이웃 노드의 중요도 학습
- 동적으로 다른 가중치 부여

**장점**:
- 🔥 **선택적 aggregation**: 중요한 이웃에 집중
- Heterogeneous graph에 강함
- 해석 가능성 (attention weights)

**단점**:
- ⚠️ **계산 복잡도 높음**: Attention 계산 오버헤드
- ⚠️ **메모리 사용량 큼**: 256K×74K 그래프에 부담
- Collaborative filtering에 LightGCN보다 성능 낮음

**적합성 (우리 프로젝트)**:
❌ 메모리 제약 (M4 16GB)
❌ Attention이 필요한 복잡한 관계가 없음

---

### 5. PinSage
**논문**: Graph Convolutional Neural Networks for Web-Scale Recommender Systems (KDD 2018)

**특징**:
- Pinterest에서 개발한 대규모 추천 시스템
- Random walk 기반 sampling
- Billions 규모 그래프에 적용

**장점**:
- 🔥 **Web-scale**: 초대규모 그래프 처리
- Visual/Text features 통합 가능

**단점**:
- ⚠️ **구현 복잡도 매우 높음**: Production-level 시스템
- Side information 필요 (우리는 rating만 있음)
- Overkill for our dataset

**적합성 (우리 프로젝트)**:
❌ 과도하게 복잡
❌ Side information 없이는 장점 활용 불가

---

### 6. KGAT (Knowledge Graph Attention Network)
**특징**:
- Knowledge Graph와 Attention 결합
- User-item-attribute heterogeneous graph

**장점**:
- Side information 활용

**단점**:
- ⚠️ **Knowledge Graph 필요**: 우리는 없음
- 복잡도 매우 높음

**적합성 (우리 프로젝트)**:
❌ Knowledge Graph 없음
❌ 불필요한 복잡도

---

### 7. GCN (Graph Convolutional Network - Vanilla)
**특징**:
- 기본 GCN 아키텍처
- Spectral-based graph convolution

**장점**:
- 이론적 기반 탄탄
- 간단한 구현

**단점**:
- Collaborative filtering에 최적화 안됨
- LightGCN/NGCF가 더 나음

**적합성 (우리 프로젝트)**:
⚠️ Baseline으로만 고려

---

## Cold-Start 문제 해결 특화 모델

### AGNN (Attribute GNN)
- Attribute graph 활용
- 하지만 우리는 user/item attributes 없음 ❌

### GPatch (Graph Neural Patching)
- Cold-start 시뮬레이션 pre-training
- 복잡도 높음 ⚠️

---

## 비교 요약표

| 모델 | 성능 | 효율성 | 메모리 | Cold-start | 구현난이도 | Sparse 적합성 | 종합 |
|------|------|--------|--------|------------|-----------|--------------|------|
| **LightGCN** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **🏆 최우선** |
| NGCF | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Baseline |
| GraphSAGE | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 고려 |
| GAT | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ |
| PinSage | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | ❌ 과도함 |
| GCN | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Baseline |

---

## 우리 데이터셋 특성 재확인

✅ **극도로 희소한 그래프** (sparsity ~0.003%)
✅ **Side information 없음** (user, item, rating만)
✅ **Cold-start 사용자 많음** (interaction ≤10)
✅ **메모리 제약** (M4 16GB)
✅ **속도 중요** (빠른 실험 iteration 필요)

---

## 최종 추천 전략

### Phase 1: 핵심 모델 구현 및 비교 (필수)
1. **LightGCN** ⭐⭐⭐⭐⭐ **(최우선 후보)**
   - 모든 면에서 우리 프로젝트에 최적
   - Sparse data에 검증된 SOTA
   - 빠른 구현 및 실험

2. **NGCF** (Baseline 비교용)
   - LightGCN보다 성능이 낮을 것으로 예상
   - 하지만 비교를 위해 구현

3. **Vanilla GCN** (Simple Baseline)
   - 가장 기본적인 베이스라인

### Phase 2: 고급 기법 실험 (선택사항, 성능 개선 필요시)
4. **GraphSAGE** 
   - Cold-start 성능 개선 기대
   - Inductive capability 활용

5. **LightGCN + Rating Prediction**
   - LightGCN에 rating regression head 추가
   - Good rating (≥4) 예측 정확도 향상

6. **Ensemble**
   - LightGCN + GraphSAGE 앙상블
   - 다양성과 정확도 동시 확보

---

## 실험 계획

### Baseline 실험
```python
models = [
    'LightGCN',      # Main model
    'NGCF',          # Comparison
    'GCN',           # Simple baseline
]
```

### 하이퍼파라미터
```python
config = {
    'embedding_dim': [32, 64, 128],
    'n_layers': [2, 3, 4],
    'learning_rate': [0.001, 0.005],
    'batch_size': [1024, 2048],
}
```

### 평가 지표
- Recall@K (primary metric)
- NDCG@K
- Precision@K
- Cold-start user 별도 평가

### 실험 순서
1. LightGCN (layer 2, emb_dim 64) - 빠른 baseline
2. Hyperparameter tuning (LightGCN)
3. NGCF 비교
4. (필요시) GraphSAGE, 앙상블 등

---

## 결론

**최종 선택: LightGCN을 메인 모델로 사용**

**근거**:
1. ✅ Sparse collaborative filtering에서 **검증된 SOTA 성능**
2. ✅ **구현 간단**, 학습 속도 빠름
3. ✅ **메모리 효율적** (M4 16GB에서 충분)
4. ✅ 2024년 현재도 **여전히 강력한 baseline**
5. ✅ 우리 데이터(rating only)와 **perfect match**

**하지만**:
- NGCF, GCN을 baseline으로 함께 구현하여 **비교 실험**
- 성능이 부족하면 GraphSAGE, 앙상블 등 시도
- **실험을 통해 검증하는 것이 핵심**

이 접근법은 단순히 LightGCN을 맹목적으로 선택하는 것이 아니라, **조사 → 비교 → 실험**을 통해 최적 모델을 찾는 과학적 방법입니다.
