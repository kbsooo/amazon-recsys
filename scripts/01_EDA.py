#%%
"""
탐색적 데이터 분석 (EDA) - Amazon RecSys GNN
데이터의 특성을 면밀히 파악하고 전처리 및 모델링 전략 수립을 위한 분석
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Mac용)
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

#%%
print("="*60)
print("1. 데이터 로드 및 기본 정보")
print("="*60)

# 데이터 로드
df = pd.read_csv('../data/amazon_train.csv')

print(f"\n데이터 형태: {df.shape}")
print(f"\n첫 5개 행:")
print(df.head())
print(f"\n데이터 타입:")
print(df.dtypes)
print(f"\n결측치 확인:")
print(df.isnull().sum())
print(f"\n기본 통계:")
print(df.describe())

#%%
print("\n" + "="*60)
print("2. 유저 및 아이템 통계")
print("="*60)

n_users = df['user'].nunique()
n_items = df['item'].nunique()
n_interactions = len(df)

print(f"\n총 유저 수: {n_users:,}")
print(f"총 아이템 수: {n_items:,}")
print(f"총 상호작용 수: {n_interactions:,}")

# 희소성 계산
sparsity = 1 - (n_interactions / (n_users * n_items))
print(f"희소성(Sparsity): {sparsity:.4%}")

# 밀도
density = n_interactions / (n_users * n_items)
print(f"밀도(Density): {density:.6f}")

#%%
print("\n" + "="*60)
print("3. 평점 분포 분석")
print("="*60)

rating_counts = df['rating'].value_counts().sort_index()
print("\n평점별 개수:")
print(rating_counts)
print(f"\n평균 평점: {df['rating'].mean():.4f}")
print(f"평점 표준편차: {df['rating'].std():.4f}")
print(f"평점 중앙값: {df['rating'].median():.1f}")

# 평점 분포 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 막대 그래프
axes[0].bar(rating_counts.index, rating_counts.values, color='steelblue', alpha=0.7)
axes[0].set_xlabel('평점', fontsize=12)
axes[0].set_ylabel('개수', fontsize=12)
axes[0].set_title('평점 분포', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# 비율로 표시
rating_pct = (rating_counts / rating_counts.sum() * 100)
axes[1].bar(rating_pct.index, rating_pct.values, color='coral', alpha=0.7)
axes[1].set_xlabel('평점', fontsize=12)
axes[1].set_ylabel('비율 (%)', fontsize=12)
axes[1].set_title('평점 비율 분포', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/01_rating_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
print("\n" + "="*60)
print("4. 유저별 상호작용 분석")
print("="*60)

user_interactions = df.groupby('user').size()
print(f"\n유저별 상호작용 수:")
print(f"  최소: {user_interactions.min()}")
print(f"  최대: {user_interactions.max()}")
print(f"  평균: {user_interactions.mean():.2f}")
print(f"  중앙값: {user_interactions.median():.0f}")
print(f"  표준편차: {user_interactions.std():.2f}")

# 분위수
print(f"\n분위수:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {q*100:.0f}%: {user_interactions.quantile(q):.0f}")

# 10개 이하 유저 수 (특수 규칙 적용 대상)
users_le_10 = (user_interactions <= 10).sum()
print(f"\n상호작용 10개 이하 유저: {users_le_10} ({users_le_10/n_users*100:.2f}%)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 히스토그램
axes[0].hist(user_interactions, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('상호작용 수', fontsize=12)
axes[0].set_ylabel('유저 수', fontsize=12)
axes[0].set_title('유저별 상호작용 분포', fontsize=14, fontweight='bold')
axes[0].axvline(user_interactions.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {user_interactions.mean():.1f}')
axes[0].axvline(10, color='orange', linestyle='--', linewidth=2, label='임계값: 10')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 로그 스케일
axes[1].hist(user_interactions, bins=50, color='green', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('상호작용 수', fontsize=12)
axes[1].set_ylabel('유저 수 (로그 스케일)', fontsize=12)
axes[1].set_title('유저별 상호작용 분포 (로그)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/02_user_interactions.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
print("\n" + "="*60)
print("5. 아이템별 상호작용 분석")
print("="*60)

item_interactions = df.groupby('item').size()
print(f"\n아이템별 상호작용 수:")
print(f"  최소: {item_interactions.min()}")
print(f"  최대: {item_interactions.max()}")
print(f"  평균: {item_interactions.mean():.2f}")
print(f"  중앙값: {item_interactions.median():.0f}")
print(f"  표준편차: {item_interactions.std():.2f}")

# 분위수
print(f"\n분위수:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    print(f"  {q*100:.0f}%: {item_interactions.quantile(q):.0f}")

# 인기도 분석
popular_threshold = item_interactions.quantile(0.8)
popular_items = (item_interactions >= popular_threshold).sum()
print(f"\n인기 아이템 (상위 20%): {popular_items} ({popular_items/n_items*100:.2f}%)")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 히스토그램
axes[0].hist(item_interactions, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('상호작용 수', fontsize=12)
axes[0].set_ylabel('아이템 수', fontsize=12)
axes[0].set_title('아이템별 상호작용 분포', fontsize=14, fontweight='bold')
axes[0].axvline(item_interactions.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {item_interactions.mean():.1f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 로그 스케일
axes[1].hist(item_interactions, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[1].set_xlabel('상호작용 수', fontsize=12)
axes[1].set_ylabel('아이템 수 (로그 스케일)', fontsize=12)
axes[1].set_title('아이템별 상호작용 분포 (로그)', fontsize=14, fontweight='bold')
axes[1].set_yscale('log')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/03_item_interactions.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
print("\n" + "="*60)
print("6. Power-law 분포 검증")
print("="*60)

# Top-K 분석
def analyze_top_k(interactions, name, k=20):
    top_k = interactions.nlargest(k)
    total = interactions.sum()
    top_k_sum = top_k.sum()
    
    print(f"\n{name} Top-{k}:")
    print(f"  상위 {k}개가 전체의 {top_k_sum/total*100:.2f}% 차지")
    print(f"  평균 상호작용: {top_k.mean():.1f}")
    
    return top_k

user_top20 = analyze_top_k(user_interactions, "유저", 20)
item_top20 = analyze_top_k(item_interactions, "아이템", 20)

# Rank vs Frequency 시각화 (Power-law)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 유저
user_sorted = user_interactions.sort_values(ascending=False).values
axes[0].loglog(range(1, len(user_sorted)+1), user_sorted, 'b.', alpha=0.5)
axes[0].set_xlabel('순위 (로그)', fontsize=12)
axes[0].set_ylabel('상호작용 수 (로그)', fontsize=12)
axes[0].set_title('유저 Power-law 분포', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# 아이템
item_sorted = item_interactions.sort_values(ascending=False).values
axes[1].loglog(range(1, len(item_sorted)+1), item_sorted, 'r.', alpha=0.5)
axes[1].set_xlabel('순위 (로그)', fontsize=12)
axes[1].set_ylabel('상호작용 수 (로그)', fontsize=12)
axes[1].set_title('아이템 Power-law 분포', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/04_power_law.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
print("\n" + "="*60)
print("7. 평점과 상호작용의 관계")
print("="*60)

# 유저별 평균 평점
user_avg_rating = df.groupby('user')['rating'].mean()
print(f"\n유저별 평균 평점:")
print(f"  평균: {user_avg_rating.mean():.4f}")
print(f"  표준편차: {user_avg_rating.std():.4f}")

# 아이템별 평균 평점
item_avg_rating = df.groupby('item')['rating'].mean()
print(f"\n아이템별 평균 평점:")
print(f"  평균: {item_avg_rating.mean():.4f}")
print(f"  표준편차: {item_avg_rating.std():.4f}")

# 상호작용 수와 평균 평점 관계
user_stats = pd.DataFrame({
    'interactions': user_interactions,
    'avg_rating': user_avg_rating
})

item_stats = pd.DataFrame({
    'interactions': item_interactions,
    'avg_rating': item_avg_rating
})

# 상관관계
print(f"\n유저: 상호작용 수 vs 평균 평점 상관계수: {user_stats.corr().iloc[0,1]:.4f}")
print(f"아이템: 상호작용 수 vs 평균 평점 상관계수: {item_stats.corr().iloc[0,1]:.4f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 유저
axes[0].scatter(user_stats['interactions'], user_stats['avg_rating'], alpha=0.3, s=10)
axes[0].set_xlabel('상호작용 수', fontsize=12)
axes[0].set_ylabel('평균 평점', fontsize=12)
axes[0].set_title('유저: 상호작용 수 vs 평균 평점', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# 아이템
axes[1].scatter(item_stats['interactions'], item_stats['avg_rating'], alpha=0.3, s=10, color='red')
axes[1].set_xlabel('상호작용 수', fontsize=12)
axes[1].set_ylabel('평균 평점', fontsize=12)
axes[1].set_title('아이템: 상호작용 수 vs 평균 평점', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/05_interactions_vs_rating.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
print("\n" + "="*60)
print("8. 추천 규칙 시뮬레이션")
print("="*60)

# 50% 규칙 시뮬레이션
def get_k_for_user(count):
    """추천 개수 계산 (README.md 규칙)"""
    if count <= 10:
        return 2
    else:
        return max(2, int(count * 0.5))

# K 분포 계산
user_k_distribution = user_interactions.apply(get_k_for_user)

print(f"\n추천 개수(K) 통계:")
print(f"  평균: {user_k_distribution.mean():.2f}")
print(f"  중앙값: {user_k_distribution.median():.0f}")
print(f"  최소: {user_k_distribution.min()}")
print(f"  최대: {user_k_distribution.max()}")

# K 분위수
print(f"\nK 분위수:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95]:
    print(f"  {q*100:.0f}%: {user_k_distribution.quantile(q):.0f}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K 분포
axes[0].hist(user_k_distribution, bins=30, color='teal', alpha=0.7, edgecolor='black')
axes[0].set_xlabel('추천 개수 (K)', fontsize=12)
axes[0].set_ylabel('유저 수', fontsize=12)
axes[0].set_title('유저별 추천 개수 분포', fontsize=14, fontweight='bold')
axes[0].axvline(user_k_distribution.mean(), color='red', linestyle='--', linewidth=2, label=f'평균: {user_k_distribution.mean():.1f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 상호작용 수 vs K
axes[1].scatter(user_interactions, user_k_distribution, alpha=0.3, s=10, color='teal')
axes[1].set_xlabel('상호작용 수', fontsize=12)
axes[1].set_ylabel('추천 개수 (K)', fontsize=12)
axes[1].set_title('상호작용 수 vs 추천 개수', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/06_recommendation_k.png', dpi=150, bbox_inches='tight')
plt.show()

#%%
print("\n" + "="*60)
print("9. 종합 요약")
print("="*60)

summary = f"""
[데이터 기본 정보]
- 전체 상호작용: {n_interactions:,}개
- 유저 수: {n_users:,}명
- 아이템 수: {n_items:,}개
- 희소성: {sparsity:.4%}

[평점 정보]
- 평균 평점: {df['rating'].mean():.2f}
- 평점 범위: {df['rating'].min():.0f} ~ {df['rating'].max():.0f}
- 가장 많은 평점: {rating_counts.idxmax()}점 ({rating_counts.max():,}개)

[유저 특성]
- 유저당 평균 상호작용: {user_interactions.mean():.1f}개
- 중앙값: {user_interactions.median():.0f}개
- 10개 이하 유저: {users_le_10}명 ({users_le_10/n_users*100:.1f}%)

[아이템 특성]
- 아이템당 평균 상호작용: {item_interactions.mean():.1f}개
- 중앙값: {item_interactions.median():.0f}개
- 상위 20% 인기 아이템: {popular_items}개

[추천 규칙 영향]
- 평균 추천 개수: {user_k_distribution.mean():.1f}개
- 총 예상 추천 수: {user_k_distribution.sum():,}개
- 특수 규칙 대상 유저(≤10): {users_le_10}명
"""

print(summary)

# 요약 저장
with open('../outputs/00_EDA_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n✅ EDA 완료! 모든 결과가 ../outputs/ 에 저장되었습니다.")
