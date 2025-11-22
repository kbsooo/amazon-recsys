#%%
"""
학습 파이프라인 - Amazon RecSys GNN
LightGCN + Rating Prediction 모델 학습
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import pickle
import time
import warnings
from pathlib import Path
import sys

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from model import LightGCN, LightGCN_Rating

warnings.filterwarnings('ignore')

# 한글 폰트
plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

#%%
print("="*60)
print("1. 환경 설정")
print("="*60)

# Random Seed
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Device: {device}")

# Hyperparameters
EMB_DIM = 64
N_LAYERS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 100
BATCH_SIZE = 2048
NUM_NEG = 4  # Negative samples per positive
HARD_NEG_RATIO = 0.5
LAMBDA_MSE = 0.5  # Rating loss weight (for CCB model)

print(f"\n하이퍼파라미터:")
print(f"  임베딩 차원: {EMB_DIM}")
print(f"  레이어 수: {N_LAYERS}")
print(f"  학습률: {LR}")
print(f"  배치 크기: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")

#%%
print("\n" + "="*60)
print("2. 데이터 로드")
print("="*60)

# 분할 데이터
train_df = pd.read_csv('../data/train_split.csv')
val_df = pd.read_csv('../data/val_split.csv')
test_df = pd.read_csv('../data/test_split.csv')

print(f"Train: {len(train_df):,}")
print(f"Val: {len(val_df):,}")
print(f"Test: {len(test_df):,}")

# 그래프 데이터
graph_data = torch.load('../data/train_graph.pt')
edge_index = graph_data['edge_index'].to(device)
cca_edge_weight = graph_data['cca_weight'].to(device)
ccb_edge_weight = graph_data['ccb_weight'].to(device)
n_users = graph_data['n_users']
n_items = graph_data['n_items']

print(f"\n그래프 정보:")
print(f"  유저 수: {n_users:,}")
print(f"  아이템 수: {n_items:,}")
print(f"  엣지 수: {edge_index.shape[1]:,}")

# Edge Sets
with open('../data/all_known_edges.pkl', 'rb') as f:
    all_known_edges = pickle.load(f)

#%%
print("\n" + "="*60)
print("3. Negative Sampling 함수")
print("="*60)

def fast_sample_negatives(batch_size, num_neg=4):
    """빠른 랜덤 negative sampling"""
    return torch.randint(0, n_items, (batch_size, num_neg), device=device)

@torch.no_grad()
def hard_negative_sampling(user_emb, item_emb, pos_users, num_neg=4, num_candidates=50):
    """
    Hard Negative Sampling
    높은 점수를 가진 negative를 선택하여 학습 효율 향상
    """
    batch_size = len(pos_users)
    candidates = torch.randint(0, n_items, (batch_size, num_candidates), device=device)
    
    user_expanded = user_emb[pos_users].unsqueeze(1)  # [B, 1, D]
    item_candidates = item_emb[candidates]  # [B, C, D]
    scores = (user_expanded * item_candidates).sum(dim=2)  # [B, C]
    
    # Top-K as hard negatives
    _, top_indices = scores.topk(num_neg, dim=1)
    hard_negs = candidates.gather(1, top_indices)
    
    return hard_negs

print("✅ Negative Sampling 함수 정의 완료")

#%%
print("\n" + "="*60)
print("4. 평가 함수")
print("="*60)

@torch.no_grad()
def evaluate_recall_ndcg(model, eval_df, edge_index, edge_weight, k_list=[20, 50]):
    """
    Recall@K 및 NDCG@K 평가
    """
    model.eval()
    u_emb, i_emb = model(edge_index, edge_weight)
    
    # User별로 그룹화
    user_groups = eval_df.groupby('user_idx')
    
    recall_at_k = {k: [] for k in k_list}
    ndcg_at_k = {k: [] for k in k_list}
    
    for user_idx, group in user_groups:
        # Ground truth items
        gt_items = set(group['item_idx'].values)
        
        # 모든 아이템에 대한 점수 계산
        user_vec = u_emb[user_idx]
        scores = (user_vec @ i_emb.t()).cpu().numpy()
        
        # Top-K 추천
        for k in k_list:
            top_k_items = np.argsort(scores)[-k:][::-1]
            
            # Recall@K
            hits = len(set(top_k_items) & gt_items)
            recall = hits / len(gt_items) if len(gt_items) > 0 else 0
            recall_at_k[k].append(recall)
            
            # NDCG@K
            dcg = sum([1 / np.log2(i + 2) if item in gt_items else 0 
                      for i, item in enumerate(top_k_items)])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(k, len(gt_items)))])
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcg_at_k[k].append(ndcg)
    
    # 평균 계산
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = np.mean(recall_at_k[k])
        metrics[f'NDCG@{k}'] = np.mean(ndcg_at_k[k])
    
    return metrics

print("✅ 평가 함수 정의 완료")

#%%
print("\n" + "="*60)
print("5. CCA 모델 학습 (Binary Recommendation)")
print("="*60)

# 모델 초기화
cca_model = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
optimizer_cca = AdamW(cca_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_cca = CosineAnnealingLR(optimizer_cca, T_max=EPOCHS)

# 학습 데이터
train_users = torch.LongTensor(train_df['user_idx'].values).to(device)
train_items = torch.LongTensor(train_df['item_idx'].values).to(device)
train_ratings = torch.FloatTensor(train_df['rating'].values).to(device)

# Rating weight (높은 평점에 더 큰 가중치)
train_weights = 0.5 + 0.15 * (train_ratings - train_ratings.mean())
train_weights = train_weights.to(device)

# 학습 이력
history_cca = {'loss': [], 'val_recall@20': [], 'val_ndcg@20': []}

print("학습 시작...")
best_val_recall = 0
patience_counter = 0
PATIENCE = 10

for epoch in range(EPOCHS):
    cca_model.train()
    perm = torch.randperm(len(train_users), device=device)
    epoch_loss = 0
    n_batches = 0
    
    for i in range(0, len(train_users), BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        pos_u = train_users[batch_idx]
        pos_i = train_items[batch_idx]
        weights = train_weights[batch_idx]
        
        # Forward
        u_emb, i_emb = cca_model(edge_index, cca_edge_weight)
        
        # Hard + Random Negative Sampling
        n_hard = int(NUM_NEG * HARD_NEG_RATIO)
        hard_negs = hard_negative_sampling(u_emb, i_emb, pos_u, num_neg=n_hard)
        rand_negs = fast_sample_negatives(len(batch_idx), NUM_NEG - n_hard)
        neg_i = torch.cat([hard_negs, rand_negs], dim=1)
        
        # Weighted BPR Loss
        pos_scores = (u_emb[pos_u] * i_emb[pos_i]).sum(dim=1)
        neg_scores = (u_emb[pos_u].unsqueeze(1) * i_emb[neg_i]).sum(dim=2)
        
        diff = pos_scores.unsqueeze(1) - neg_scores
        loss_per_sample = -torch.log(torch.sigmoid(diff) + 1e-8).mean(dim=1)
        loss = (loss_per_sample * weights).mean()
        
        # Backward
        optimizer_cca.zero_grad()
        loss.backward()
        optimizer_cca.step()
        
        epoch_loss += loss.item()
        n_batches += 1
    
    scheduler_cca.step()
    avg_loss = epoch_loss / n_batches
    history_cca['loss'].append(avg_loss)
    
    # Validation (매 5 epoch)
    if (epoch + 1) % 5 == 0:
        val_metrics = evaluate_recall_ndcg(cca_model, val_df, edge_index, cca_edge_weight, k_list=[20])
        history_cca['val_recall@20'].append(val_metrics['Recall@20'])
        history_cca['val_ndcg@20'].append(val_metrics['NDCG@20'])
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"Val Recall@20: {val_metrics['Recall@20']:.4f} | "
              f"Val NDCG@20: {val_metrics['NDCG@20']:.4f}")
        
        # Early Stopping
        if val_metrics['Recall@20'] > best_val_recall:
            best_val_recall = val_metrics['Recall@20']
            patience_counter = 0
            # 모델 저장
            torch.save(cca_model.state_dict(), '../models/cca_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ CCA 모델 학습 완료 (Best Val Recall@20: {best_val_recall:.4f})")

#%%
print("\n" + "="*60)
print("6. CCB 모델 학습 (Rating Prediction + BPR)")
print("="*60)

# 모델 초기화
ccb_model = LightGCN_Rating(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
optimizer_ccb = AdamW(ccb_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_ccb = CosineAnnealingLR(optimizer_ccb, T_max=EPOCHS)

# 학습 이력
history_ccb = {'loss': [], 'bpr_loss': [], 'mse_loss': [], 'val_rmse': []}

print("학습 시작...")
best_val_rmse = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    ccb_model.train()
    perm = torch.randperm(len(train_users), device=device)
    epoch_loss = 0
    epoch_bpr = 0
    epoch_mse = 0
    n_batches = 0
    
    for i in range(0, len(train_users), BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        pos_u = train_users[batch_idx]
        pos_i = train_items[batch_idx]
        pos_r = train_ratings[batch_idx]
        
        # Forward
        u_emb, i_emb = ccb_model(edge_index, ccb_edge_weight)
        
        # Negative Sampling (BPR)
        neg_i = fast_sample_negatives(len(batch_idx), NUM_NEG)
        
        # BPR Loss
        pos_scores = (u_emb[pos_u] * i_emb[pos_i]).sum(dim=1)
        neg_scores = (u_emb[pos_u].unsqueeze(1) * i_emb[neg_i]).sum(dim=2)
        loss_bpr = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-8).mean()
        
        # MSE Loss (Rating Prediction)
        pred_r = ccb_model.predict_rating(pos_u, pos_i, edge_index, ccb_edge_weight)
        loss_mse = F.mse_loss(pred_r, pos_r)
        
        # Total Loss
        loss = loss_bpr + LAMBDA_MSE * loss_mse
        
        # Backward
        optimizer_ccb.zero_grad()
        loss.backward()
        optimizer_ccb.step()
        
        epoch_loss += loss.item()
        epoch_bpr += loss_bpr.item()
        epoch_mse += loss_mse.item()
        n_batches += 1
    
    scheduler_ccb.step()
    avg_loss = epoch_loss / n_batches
    avg_bpr = epoch_bpr / n_batches
    avg_mse = epoch_mse / n_batches
    
    history_ccb['loss'].append(avg_loss)
    history_ccb['bpr_loss'].append(avg_bpr)
    history_ccb['mse_loss'].append(avg_mse)
    
    # Validation (매 5 epoch)
    if (epoch + 1) % 5 == 0:
        ccb_model.eval()
        with torch.no_grad():
            val_u = torch.LongTensor(val_df['user_idx'].values).to(device)
            val_i = torch.LongTensor(val_df['item_idx'].values).to(device)
            val_r = torch.FloatTensor(val_df['rating'].values).to(device)
            
            val_pred = ccb_model.predict_rating(val_u, val_i, edge_index, ccb_edge_weight)
            val_rmse = torch.sqrt(F.mse_loss(val_pred, val_r)).item()
            history_ccb['val_rmse'].append(val_rmse)
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {avg_loss:.4f} | "
              f"BPR: {avg_bpr:.4f} | MSE: {avg_mse:.4f} | Val RMSE: {val_rmse:.4f}")
        
        # Early Stopping
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(ccb_model.state_dict(), '../models/ccb_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ CCB 모델 학습 완료 (Best Val RMSE: {best_val_rmse:.4f})")

#%%
print("\n" + "="*60)
print("7. 학습 곡선 시각화")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# CCA Loss
axes[0, 0].plot(history_cca['loss'], label='Train Loss', color='blue')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('CCA Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# CCA Metrics
if history_cca['val_recall@20']:
    x = np.arange(5, len(history_cca['loss']) + 1, 5)[:len(history_cca['val_recall@20'])]
    axes[0, 1].plot(x, history_cca['val_recall@20'], label='Recall@20', marker='o')
    axes[0, 1].plot(x, history_cca['val_ndcg@20'], label='NDCG@20', marker='s')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('CCA Validation Metrics')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

# CCB Losses
axes[1, 0].plot(history_ccb['bpr_loss'], label='BPR Loss', color='orange')
axes[1, 0].plot(history_ccb['mse_loss'], label='MSE Loss', color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('CCB Training Losses')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# CCB RMSE
if history_ccb['val_rmse']:
    x = np.arange(5, len(history_ccb['loss']) + 1, 5)[:len(history_ccb['val_rmse'])]
    axes[1, 1].plot(x, history_ccb['val_rmse'], label='Val RMSE', marker='o', color='red')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('RMSE')
    axes[1, 1].set_title('CCB Validation RMSE')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/training_curves.png', dpi=150)
plt.show()

print("✅ 학습 곡선 저장 완료 (outputs/training_curves.png)")

#%%
print("\n" + "="*60)
print("8. 최종 모델 저장")
print("="*60)

# 최종 모델 로드 (Best validation)
cca_model.load_state_dict(torch.load('../models/cca_best.pt'))
ccb_model.load_state_dict(torch.load('../models/ccb_best.pt'))

# 전체 저장
torch.save({
    'cca_state_dict': cca_model.state_dict(),
    'ccb_state_dict': ccb_model.state_dict(),
    'config': {
        'n_users': n_users,
        'n_items': n_items,
        'emb_dim': EMB_DIM,
        'n_layers': N_LAYERS
    },
    'history': {
        'cca': history_cca,
        'ccb': history_ccb
    }
}, '../models/ensemble_model.pt')

print("✅ 모델 저장 완료:")
print("  - models/cca_best.pt")
print("  - models/ccb_best.pt")
print("  - models/ensemble_model.pt")

print("\n✅ 학습 파이프라인 실행 완료!")

# %%
