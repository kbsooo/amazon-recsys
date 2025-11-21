# Amazon 추천 시스템 - 최종 통합 노트북

이 노트북은 전체 파이프라인을 통합합니다:
1. 데이터 로딩 및 전처리
2. EDA (핵심 인사이트만)
3. GNN 모델 구현 및 비교
4. 학습 및 평가
5. 추천 생성
6. 결과 분석 및 시각화

## README

이 프로젝트의 목표:
- Amazon 사용자-아이템 interaction 데이터로 GNN 기반 추천 시스템 구축
- **특별한 평가 규칙**: 사용자별 맞춤 추천 개수 (기존 interaction의 50% 이하)
- Cold-start 사용자 (≤10 interactions)는 무조건 2개 추천
- LightGCN, NGCF, SimpleGCN 모델 비교 실험
