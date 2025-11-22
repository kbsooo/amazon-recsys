#%%
"""
간소화된 CCA+CCB 앙상블 평가 스크립트
빠른 비교를 위해 샘플 유저만 평가
"""
import numpy as np
import pandas as pd
import torch
import sys
from pathlib import Path

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

data_dir = Path(__file__).parent.parent / 'data'
models_dir = Path(__file__).parent.parent / 'models'

# 데이터 로드
test_df = pd.read_csv(data_dir / 'test_split.csv')

# 그래프 데이터
graph_data = torch.load(data_dir / 'train_graph.pt', map_location=device, weights_only=False)
edge_index = graph_data['edge_index'].to(device)
cca_weight = graph_data['cca_weight'].to(device)
ccb_weight = graph_data['ccb_weight'].to(device)
n_users = graph_data['n_users']
n_items = graph_data['n_items']

# 모델 로드
EMB_DIM = 64
N_LAYERS = 3

cca = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
cca.load_state_dict(torch.load(models_dir / 'cca_best.pt', map_location=device))
cca.eval()

ccb = LightGCN_Rating(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
ccb.load_state_dict(torch.load(models_dir / 'ccb_best.pt', map_location=device))
ccb.eval()

print("✅ Models Loaded")

# 간단한 평가: Top-20 Recall/NDCG
def eval_top20(alpha=0.5, max_users=500):
    """Alpha: CCA 가중치 (1-alpha = CCB 가중치)"""
    user_groups = test_df.groupby('user_idx')
    recalls, ndcgs = [], []
    
    # Pre-compute embeddings for efficiency
    with torch.no_grad():
        cca_u, cca_i = cca(edge_index, cca_weight)
        ccb_u, ccb_i = ccb(edge_index, ccb_weight)
    
    # Normalization range (단순화)
    CCA_MIN, CCA_MAX = -0.5, 0.5
    CCB_MIN, CCB_MAX = 0.5, 5.0
    
    user_count = 0
    for user_idx, group in user_groups:
        if user_count >= max_users: break
        user_count += 1
        
        gt_items = set(group['item_idx'].values)
        if len(gt_items) == 0: continue
        
        # 모든 아이템에 대해 점수 계산 (샘플링 없이)
        # all_items = torch.arange(n_items, device=device) # Not needed if we use broadcasting
        
        with torch.no_grad():
            # CCA Score
            cca_s = (cca_u[user_idx] * cca_i).sum(dim=1).cpu().numpy()
            
            # CCB Rating (Optimized)
            # interaction = ccb_u[user_idx] * ccb_i  # Broadcasting: (emb_dim) * (n_items, emb_dim) -> (n_items, emb_dim)
            interaction = ccb_u[user_idx].unsqueeze(0) * ccb_i
            rating_logit = ccb.rating_mlp(interaction).squeeze(-1)
            ccb_r = (torch.sigmoid(rating_logit) * 4.5 + 0.5).cpu().numpy()
        
        # Normalize
        cca_n = np.clip((cca_s - CCA_MIN) / (CCA_MAX - CCA_MIN + 1e-8), 0, 1)
        ccb_n = np.clip((ccb_r - CCB_MIN) / (CCB_MAX - CCB_MIN + 1e-8), 0, 1)
        
        # Ensemble
        scores = alpha * cca_n + (1 - alpha) * ccb_n
        
        # Top-20
        top20 = set(np.argsort(scores)[-20:])
        
        # Recall
        hits = len(top20 & gt_items)
        recalls.append(hits / len(gt_items))
        
        # NDCG
        dcg = sum([1/np.log2(i+2) if item in gt_items else 0 for i, item in enumerate(sorted(top20, key=lambda x: -scores[x]))])
        idcg = sum([1/np.log2(i+2) for i in range(min(20, len(gt_items)))])
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(recalls), np.mean(ndcgs)

print("\n=== CCA+CCB Ensemble Evaluation ===")
print("Testing Alpha values...")

best_alpha = 0.5
best_recall = 0

for alpha in [0.3, 0.5, 0.7, 1.0]:
    recall, ndcg = eval_top20(alpha, max_users=200)
    print(f"Alpha {alpha:.1f}: Recall@20={recall:.4f}, NDCG@20={ndcg:.4f}")
    if recall > best_recall:
        best_recall = recall
        best_alpha = alpha

print(f"\n최적 Alpha: {best_alpha}")
print(f"\n=== 최종 비교 (Test Set 500명 샘플) ===")
final_recall, final_ndcg = eval_top20(best_alpha, max_users=500)

print(f"\nCCA+CCB Ensemble (Alpha={best_alpha}):")
print(f"  Recall@20: {final_recall:.4f}")
print(f"  NDCG@20:   {final_ndcg:.4f}")

print(f"\nSimGCL (Ours):")
print(f"  Recall@20: 0.4931")
print(f"  NDCG@20:   0.2389")

diff_r = final_recall - 0.4931
diff_n = final_ndcg - 0.2389

print(f"\n차이 (Ensemble - SimGCL):")
print(f"  Recall: {diff_r:+.4f} ({diff_r/0.4931*100:+.1f}%)")
print(f"  NDCG:   {diff_n:+.4f} ({diff_n/0.2389*100:+.1f}%)")

if diff_r < 0:
    print("\n✅ SimGCL이 CCA+CCB 앙상블보다 우수합니다!")
else:
    print("\n⚠️ CCA+CCB 앙상블이 더 좋은 성능을 보입니다.")
