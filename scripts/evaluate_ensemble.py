#%%
"""
CCA + CCB 앙상블 평가 스크립트
목적: 기존 베이스라인(CCA+CCB)의 성능을 측정하여 SimGCL과 비교.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from model import LightGCN, LightGCN_Rating

# 환경 설정
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# 데이터 경로
data_dir = Path(__file__).parent.parent / 'data'
models_dir = Path(__file__).parent.parent / 'models'

# 데이터 로드
val_df = pd.read_csv(data_dir / 'val_split.csv')
test_df = pd.read_csv(data_dir / 'test_split.csv')

# 그래프 데이터 로드
graph_data = torch.load(data_dir / 'train_graph.pt')
edge_index = graph_data['edge_index'].to(device)
cca_edge_weight = graph_data['cca_weight'].to(device)
ccb_edge_weight = graph_data['ccb_weight'].to(device)
n_users = graph_data['n_users']
n_items = graph_data['n_items']

# 모델 설정
EMB_DIM = 64
N_LAYERS = 3

# 모델 로드
print("Loading Models...")
cca_model = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
try:
    cca_model.load_state_dict(torch.load(models_dir / 'cca_best.pt', map_location=device))
    print("✅ CCA Model Loaded")
except FileNotFoundError:
    print("❌ CCA Model Not Found! (Please train CCA first)")
    sys.exit(1)

ccb_model = LightGCN_Rating(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
try:
    ccb_model.load_state_dict(torch.load(models_dir / 'ccb_best.pt', map_location=device))
    print("✅ CCB Model Loaded")
except FileNotFoundError:
    print("❌ CCB Model Not Found! (Please train CCB first)")
    sys.exit(1)

cca_model.eval()
ccb_model.eval()

# Normalization 함수
def normalize(val, v_min, v_max):
    return np.clip((val - v_min) / (v_max - v_min + 1e-8), 0, 1)

# 평가 함수 (Recall@K, NDCG@K)
def evaluate_ensemble(alpha, eval_df, k=20):
    user_groups = eval_df.groupby('user_idx')
    recall_list = []
    ndcg_list = []
    
    # Pre-compute embeddings
    with torch.no_grad():
        cca_u, cca_i = cca_model(edge_index, cca_edge_weight)
        ccb_u, ccb_i = ccb_model(edge_index, ccb_edge_weight)
    
    # Score Range Estimation (Sample)
    sample_indices = torch.randint(0, n_users, (1000,), device=device)
    sample_items = torch.randint(0, n_items, (1000,), device=device)
    
    with torch.no_grad():
        cca_s = (cca_u[sample_indices] * cca_i[sample_items]).sum(dim=1).cpu().numpy()
        ccb_r = ccb_model.predict_rating(sample_indices, sample_items, edge_index, ccb_edge_weight).cpu().numpy()
    
    CCA_MIN, CCA_MAX = cca_s.min(), cca_s.max()
    CCB_MIN, CCB_MAX = 0.5, 5.0 # Rating range
    
    for user_idx, group in user_groups:
        gt_items = set(group['item_idx'].values)
        if len(gt_items) == 0: continue
        
        # Candidate Selection (All items is too slow, sample negatives + positives)
        # For fair comparison, we should rank ALL items or a large set.
        # Here we use a simplified approach: Rank all items for the user (if feasible) or large sample.
        # Given 74k items, ranking all is slow. Let's use 1000 candidates (Positives + Random Negatives)
        
        # Positive items
        pos_items = list(gt_items)
        
        # Negative items (Random sample to make total 1000)
        num_neg = 1000 - len(pos_items)
        if num_neg > 0:
            neg_items = np.random.randint(0, n_items, num_neg)
            candidates = np.array(pos_items + list(neg_items))
        else:
            candidates = np.array(pos_items)
            
        cand_tensor = torch.LongTensor(candidates).to(device)
        user_tensor = torch.LongTensor([user_idx] * len(candidates)).to(device)
        
        with torch.no_grad():
            # CCA Score
            cca_scores = (cca_u[user_idx] * cca_i[cand_tensor]).sum(dim=1).cpu().numpy()
            # CCB Rating
            # Note: predicting rating for 1000 items might be slow if using full MLP forward.
            # Optimization: Pre-compute embeddings and do dot product + MLP only for interaction
            # But LightGCN_Rating structure requires edge_index for forward.
            # We use the model's predict_rating method which does forward internally.
            # To speed up, we could pass pre-computed embeddings if the class supported it.
            # For now, we just call it. It might be slow.
            ccb_ratings = ccb_model.predict_rating(user_tensor, cand_tensor, edge_index, ccb_edge_weight).cpu().numpy()
            
        # Normalize
        cca_norm = normalize(cca_scores, CCA_MIN, CCA_MAX)
        ccb_norm = normalize(ccb_ratings, CCB_MIN, CCB_MAX)
        
        # Ensemble
        final_scores = alpha * cca_norm + (1 - alpha) * ccb_norm
        
        # Top-K
        top_k_indices = np.argsort(final_scores)[-k:][::-1]
        top_k_items = candidates[top_k_indices]
        
        # Metrics
        hits = len(set(top_k_items) & gt_items)
        recall = hits / len(gt_items)
        recall_list.append(recall)
        
        dcg = sum([1 / np.log2(i + 2) if item in gt_items else 0 for i, item in enumerate(top_k_items)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(gt_items)))])
        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_list.append(ndcg)
        
    return np.mean(recall_list), np.mean(ndcg_list)

print("\n--- Alpha Tuning on Validation Set ---")
best_alpha = 0.5
best_val_recall = 0

# Grid Search
for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    # Note: Using a subset of users for faster tuning
    val_subset = val_df[val_df['user_idx'] < 1000] # First 1000 users
    recall, ndcg = evaluate_ensemble(alpha, val_subset)
    print(f"Alpha {alpha:.1f}: Recall@20={recall:.4f}, NDCG@20={ndcg:.4f}")
    
    if recall > best_val_recall:
        best_val_recall = recall
        best_alpha = alpha

print(f"\nBest Alpha: {best_alpha} (Val Recall: {best_val_recall:.4f})")

print("\n--- Final Evaluation on Test Set ---")
# Use best alpha on full test set (or subset if too slow)
# Testing on first 1000 users for speed demonstration
test_subset = test_df[test_df['user_idx'] < 1000] 
test_recall, test_ndcg = evaluate_ensemble(best_alpha, test_subset)

print(f"CCA+CCB Ensemble (Alpha={best_alpha}):")
print(f"  Recall@20: {test_recall:.4f}")
print(f"  NDCG@20:   {test_ndcg:.4f}")

print("\n--- Comparison with SimGCL ---")
print(f"SimGCL (Ours):")
print(f"  Recall@20: 0.4931")
print(f"  NDCG@20:   0.2389")

diff_recall = test_recall - 0.4931
diff_ndcg = test_ndcg - 0.2389

print(f"\nDifference (Ensemble - SimGCL):")
print(f"  Recall: {diff_recall:+.4f}")
print(f"  NDCG:   {diff_ndcg:+.4f}")

if diff_recall < 0:
    print("\n✅ SimGCL outperforms CCA+CCB Ensemble!")
else:
    print("\n⚠️ CCA+CCB Ensemble is better.")
