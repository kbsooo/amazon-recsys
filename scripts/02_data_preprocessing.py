#%%
"""
데이터 로더 및 전처리 - Amazon RecSys GNN
Train/Val/Test 분할, 그래프 생성, 정규화
"""
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, Tuple, Set
import warnings

warnings.filterwarnings('ignore')

# Random Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

#%%
print("="*60)
print("1. 데이터 로드 및 ID 매핑")
print("="*60)

# 데이터 로드
df = pd.read_csv('../data/amazon_train.csv')
print(f"원본 데이터: {df.shape}")

# ID 매핑 (정렬된 순서로)
user2idx = {u: i for i, u in enumerate(sorted(df['user'].unique()))}
item2idx = {it: i for i, it in enumerate(sorted(df['item'].unique()))}
idx2user = {i: u for u, i in user2idx.items()}
idx2item = {i: it for it, i in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)

# 인덱스 추가
df['user_idx'] = df['user'].map(user2idx)
df['item_idx'] = df['item'].map(item2idx)

print(f"유저 수: {n_users:,}")
print(f"아이템 수: {n_items:,}")
print(f"상호작용 수: {len(df):,}")

# ID 매핑 저장
print("\nID 매핑 저장 중...")
import pickle
with open('../data/user2idx.pkl', 'wb') as f:
    pickle.dump(user2idx, f)
with open('../data/item2idx.pkl', 'wb') as f:
    pickle.dump(item2idx, f)
with open('../data/idx2user.pkl', 'wb') as f:
    pickle.dump(idx2user, f)
with open('../data/idx2item.pkl', 'wb') as f:
    pickle.dump(idx2item, f)
print("✅ ID 매핑 저장 완료")

#%%
print("\n" + "="*60)
print("2. Good/Bad Rating 분리")
print("="*60)

# Good Rating Threshold (EDA 기반)
GOOD_RATING_THRESHOLD = 4.0

df['is_good'] = df['rating'] >= GOOD_RATING_THRESHOLD
good_df = df[df['is_good']].copy()
bad_df = df[~df['is_good']].copy()

print(f"Good Ratings (≥{GOOD_RATING_THRESHOLD}): {len(good_df):,} ({len(good_df)/len(df)*100:.1f}%)")
print(f"Bad Ratings (<{GOOD_RATING_THRESHOLD}): {len(bad_df):,} ({len(bad_df)/len(df)*100:.1f}%)")

#%%
print("\n" + "="*60)
print("3. Train/Val/Test 분할 (User-wise Stratified)")
print("="*60)

# 전략: Bad ratings는 모두 train으로, Good ratings만 분할
train_data, val_data, test_data = [], [], []

for user_idx in range(n_users):
    user_df = df[df['user_idx'] == user_idx]
    
    # Bad ratings는 무조건 train (구조 학습용)
    user_bad = user_df[~user_df['is_good']]
    if len(user_bad) > 0:
        train_data.append(user_bad)
    
    # Good ratings 분할
    user_good = user_df[user_df['is_good']]
    n_good = len(user_good)
    
    if n_good >= 3:
        # 70/15/15 분할
        user_good = user_good.sample(frac=1, random_state=SEED).reset_index(drop=True)
        tr_end = int(0.7 * n_good)
        val_end = tr_end + int(0.15 * n_good)
        tr_end = max(1, tr_end)
        val_end = max(tr_end + 1, val_end)
        
        train_data.append(user_good.iloc[:tr_end])
        val_data.append(user_good.iloc[tr_end:val_end])
        test_data.append(user_good.iloc[val_end:])
    elif n_good == 2:
        # 1개씩 train/val
        user_good = user_good.sample(frac=1, random_state=SEED).reset_index(drop=True)
        train_data.append(user_good.iloc[:1])
        val_data.append(user_good.iloc[1:])
    elif n_good == 1:
        # 전부 train
        train_data.append(user_good)

train_df = pd.concat(train_data, ignore_index=True)
val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()

print(f"Train: {len(train_df):,}")
print(f"Val: {len(val_df):,}")
print(f"Test: {len(test_df):,}")
print(f"Total: {len(train_df) + len(val_df) + len(test_df):,}")

# 분할 데이터 저장
train_df.to_csv('../data/train_split.csv', index=False)
val_df.to_csv('../data/val_split.csv', index=False)
test_df.to_csv('../data/test_split.csv', index=False)
print("\n✅ 분할 데이터 저장 완료")

#%%
print("\n" + "="*60)
print("4. 유저별 Train Items 및 K 계산")
print("="*60)

# 유저별 train items (추론 시 필터링용)
user_train_items = defaultdict(set)
for u, i in zip(train_df['user_idx'].values, train_df['item_idx'].values):
    user_train_items[int(u)].add(int(i))

# 유저별 상호작용 수
user_interaction_count = train_df.groupby('user_idx').size().to_dict()

# K 계산 (README.md 규칙)
MAX_K = 100

def get_k_for_user(count):
    """추천 개수 계산"""
    if count <= 10:
        return 2
    else:
        k = max(2, int(count * 0.5))
        return min(k, MAX_K)

user_k = {u: get_k_for_user(c) for u, c in user_interaction_count.items()}

print(f"유저별 K 통계:")
k_values = list(user_k.values())
print(f"  평균: {np.mean(k_values):.2f}")
print(f"  중앙값: {np.median(k_values):.0f}")
print(f"  최소: {np.min(k_values)}")
print(f"  최대: {np.max(k_values)}")

# 저장
with open('../data/user_train_items.pkl', 'wb') as f:
    pickle.dump(dict(user_train_items), f)
with open('../data/user_k.pkl', 'wb') as f:
    pickle.dump(user_k, f)
print("\n✅ User metadata 저장 완료")

#%%
print("\n" + "="*60)
print("5. 그래프 생성 (Edge Index)")
print("="*60)

def build_edge_index_and_weights(df, n_users, n_items):
    """
    Edge Index 및 Weights 생성
    
    Returns:
        edge_index: [2, num_edges] - 양방향 엣지
        cca_weight: Symmetric Normalization (Unweighted)
        ccb_weight: Rating-weighted Normalization
    """
    users = df['user_idx'].values
    items = df['item_idx'].values
    ratings = df['rating'].values
    
    # Edge Index 생성 (양방향)
    edge_u2i = np.array([users, items + n_users])  # user -> item
    edge_i2u = np.array([items + n_users, users])  # item -> user
    edge_index = np.concatenate([edge_u2i, edge_i2u], axis=1)
    
    # Degree 계산
    num_nodes = n_users + n_items
    edge_index_tensor = torch.LongTensor(edge_index)
    deg = torch.zeros(num_nodes).scatter_add(
        0, edge_index_tensor[0], torch.ones(edge_index_tensor.shape[1])
    )
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # CCA Weight: Symmetric Normalization (Unweighted)
    cca_weight = deg_inv_sqrt[edge_index_tensor[0]] * deg_inv_sqrt[edge_index_tensor[1]]
    
    # CCB Weight: Rating-weighted
    # Formula: weight = base + rating * scale
    rating_factors = 0.4 + 0.15 * ratings  # 평점 1~5 -> 0.55~1.15
    rating_factors_both = np.concatenate([rating_factors, rating_factors])
    ccb_weight = cca_weight * torch.FloatTensor(rating_factors_both)
    
    return edge_index_tensor, cca_weight, ccb_weight

# Train Graph 생성
train_edge_index, train_cca_weight, train_ccb_weight = build_edge_index_and_weights(
    train_df, n_users, n_items
)

print(f"Train Graph:")
print(f"  노드 수: {n_users + n_items:,}")
print(f"  엣지 수: {train_edge_index.shape[1]:,}")
print(f"  CCA Weight 범위: [{train_cca_weight.min():.4f}, {train_cca_weight.max():.4f}]")
print(f"  CCB Weight 범위: [{train_ccb_weight.min():.4f}, {train_ccb_weight.max():.4f}]")

# 저장
torch.save({
    'edge_index': train_edge_index,
    'cca_weight': train_cca_weight,
    'ccb_weight': train_ccb_weight,
    'n_users': n_users,
    'n_items': n_items
}, '../data/train_graph.pt')
print("\n✅ Train Graph 저장 완료")

#%%
print("\n" + "="*60)
print("6. Validation/Test Edge Sets (Negative Sampling용)")
print("="*60)

# Val/Test 엣지 집합
val_edges = set(zip(val_df['user_idx'].values, val_df['item_idx'].values))
test_edges = set(zip(test_df['user_idx'].values, test_df['item_idx'].values))
train_edges = set(zip(train_df['user_idx'].values, train_df['item_idx'].values))

print(f"Train Edges: {len(train_edges):,}")
print(f"Val Edges: {len(val_edges):,}")
print(f"Test Edges: {len(test_edges):,}")

# 모든 known edges (negative sampling 시 제외용)
all_known_edges = train_edges | val_edges | test_edges
print(f"Total Known Edges: {len(all_known_edges):,}")

# 저장
with open('../data/train_edges.pkl', 'wb') as f:
    pickle.dump(train_edges, f)
with open('../data/val_edges.pkl', 'wb') as f:
    pickle.dump(val_edges, f)
with open('../data/test_edges.pkl', 'wb') as f:
    pickle.dump(test_edges, f)
with open('../data/all_known_edges.pkl', 'wb') as f:
    pickle.dump(all_known_edges, f)
print("\n✅ Edge Sets 저장 완료")

#%%
print("\n" + "="*60)
print("7. 데이터 요약")
print("="*60)

summary = f"""
[데이터 분할]
- Train: {len(train_df):,} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)
- Val: {len(val_df):,} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)
- Test: {len(test_df):,} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)

[그래프 정보]
- 노드: User {n_users:,} + Item {n_items:,} = {n_users+n_items:,}
- Train Edges: {train_edge_index.shape[1]:,} (양방향)
- Val Edges: {len(val_edges):,}
- Test Edges: {len(test_edges):,}

[추천 규칙]
- 평균 K: {np.mean(k_values):.2f}
- K=2 유저: {sum(1 for k in k_values if k == 2):,}명
- K>2 유저: {sum(1 for k in k_values if k > 2):,}명

[저장된 파일]
- data/train_split.csv, val_split.csv, test_split.csv
- data/train_graph.pt
- data/user2idx.pkl, item2idx.pkl (및 역매핑)
- data/user_train_items.pkl, user_k.pkl
- data/*_edges.pkl
"""

print(summary)

with open('../outputs/01_data_preprocessing_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n✅ 데이터 전처리 완료!")
