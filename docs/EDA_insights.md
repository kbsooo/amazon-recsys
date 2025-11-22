# EDA 주요 발견사항 및 전략 수립

## 📊 데이터 특성 분석 결과

### 1. 극도로 희소한 데이터
- **희소성**: 99.997% (거의 모든 user-item 조합이 비어있음)
- **밀도**: 0.00003
- **임의적 추천의 어려움**: Cold-start 문제가 심각

### 2. 롱테일 분포 (Long-tail Distribution)
- **유저**:
  - 평균 상호작용: 2.2개
  - 중앙값: 1개 (대부분의 유저가 극소수 아이템만 구매)
  - 상위 1% 유저만 17개 이상 구매
  
- **아이템**:
  - 평균 상호작용: 7.7개
  - 중앙값: 2개
  - 극소수 인기 아이템이 대부분의 상호작용 차지

### 3. 평점 분포의 편향
- **5점 편중**: 전체의 63.9%가 5점
- **긍정 편향**: 4~5점이 78.1%
- **평균 평점**: 4.18점
- **시사점**: Binary Implicit Feedback으로 처리하거나, 평점 가중치 필요

### 4. 특수 규칙 적용 대상
- **10개 이하 유저**: 249,655명 (97.5%)
  - 거의 모든 유저가 "무조건 2개 추천" 규칙 적용 대상
  - 이는 모델 설계에 큰 영향을 미침

## 🎯 모델링 전략

### 1. 데이터 전처리 전략
```
✅ Binary Implicit Feedback 사용
   - 평점 ≥ 4.0 → Positive (Good)
   - 평점 < 4.0 → Negative (사용하지 않거나 구조 학습에만 활용)
   
✅ Stratified Split
   - User-wise split으로 각 유저의 history 보존
   - Good ratings만 test로, Bad ratings는 train에 포함 (prev.py 방식)
   
✅ 그래프 구성
   - Unweighted Graph: 구조 학습 (CCA)
   - Weighted Graph: 평점 가중치 적용 (CCB)
```

### 2. 모델 설계 방향
```
🔹 LightGCN 기반 (가벼우면서도 효과적)
   - 희소 데이터에 강함
   - Over-smoothing 방지 (레이어 2~3개 권장)
   
🔹 Ensemble 전략 (prev.py 참고)
   - CCA: 구조 기반 Binary Prediction
   - CCB: 평점 예측 모델
   - Alpha/Beta 가중 결합
   
🔹 Negative Sampling
   - Hard Negative Sampling (BPR Loss)
   - 높은 평점에 더 큰 가중치
```

### 3. 평가 및 추론 전략
```
📌 평가 메트릭
   - Recall@K, NDCG@K
   - Custom "O-Ratio" (추천 비율 모니터링)
   
📌 추론 로직
   1. 모델 점수 계산
   2. Threshold 필터링
   3. Top-K 선택
   4. 특수 규칙 적용:
      - 10개 이하 유저 → 무조건 2개
      - 나머지 → 최대 50%까지
```

## 💡 Breakthrough Ideas (향후 실험)

### 아이디어 1: User/Item Bias 명시적 모델링
- 평점 편향이 심하므로, User/Item별 bias term 추가하여 calibration

### 아이디어 2: Contrastive Learning
- Self-supervised learning으로 희소성 극복
- Augmentation: Edge Dropout, Feature Masking

### 아이디어 3: Multi-task Learning
- Binary Recommendation + Rating Prediction 동시 학습
- Auxiliary task로 일반화 성능 향상

### 아이디어 4: Adaptive K
- 유저별 상호작용 패턴 학습하여 K 동적 조정
- 단순 50% 규칙보다 정교화

### 아이디어 5: Graph Attention
- 모든 이웃을 동등하게 취급하지 않고, 중요도 학습
- 인기 아이템 편향 완화

## 📋 다음 단계

1. **데이터 전처리 파이프라인 구축**
2. **LightGCN 베이스라인 구현**
3. **앙상블 모델 구현**
4. **학습 및 검증**
5. **실험을 통한 최적화**
