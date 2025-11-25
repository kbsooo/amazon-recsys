#%% [markdown]
# # Amazon RecSys - 통합 파이프라인 (PyTorch)
# 
# 전처리 → EASE 학습 → SimGCL 학습 → 앙상블 → 평가를 모두 포함한 완전한 파이프라인입니다.

#%% [code]
import os
import sys
import gc
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from collections import defaultdict

# Add src to path
sys.path.append(os.path.abspath('../src'))

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Paths
DATA_DIR = '/kaggle/input/amazon' if os.path.exists('/kaggle/input/amazon') else 'data'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 
                      'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"🚀 Device: {device}")

#%% [markdown]
# ## Part 1: 데이터 로드 및 준비

#%% [code]
print("\n" + "="*60)
print("📊 데이터 로드")
print("="*60)

train_df = pd.read_csv(f'{DATA_DIR}/train_split.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_split.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test_split.csv')

with open(f'{DATA_DIR}/user2idx.pkl', 'rb') as f:
    user2idx=pickle.load(f)
with open(f'{DATA_DIR}/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)
with open(f'{DATA_DIR}/user_k.pkl', 'rb') as f:
    user_k = pickle.load(f)
with open(f'{DATA_DIR}/user_train_items.pkl', 'rb') as f:
    user_train_items = pickle.load(f)

# Reverse mappings
idx2user = {v: k for k, v in user2idx.items()}
idx2item = {v: k for k, v in item2idx.items()}

n_users = len(user2idx)
n_items = len(item2idx)

print(f"Users: {n_users}, Items: {n_items}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Load Graph
graph_data = torch.load(f'{DATA_DIR}/train_graph.pt', map_location=device, weights_only=False)
edge_index = graph_data['edge_index'].to(device)
edge_weight = graph_data['cca_weight'].to(device)

#%% [markdown]
# ## Part 2: EASE 모델 학습

#%% [code]
print("\n" + "="*60)
print("🔧 EASE 모델 학습")
print("="*60)

LAMBDA_EASE = 500.0

# Interaction Matrix (Train + Val)
full_train_df = pd.concat([train_df, val_df])
rows = full_train_df['user_idx'].values
cols = full_train_df['item_idx'].values
data = np.ones(len(rows), dtype=np.float32)
X = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

# Gram Matrix
print("Computing Gram Matrix...")
start_time = time.time()
G = X.T.dot(X)
print(f"⏱️ Gram Matrix: {time.time() - start_time:.2f}s")

# Convert to dense
print("Converting to dense...")
G_dense = torch.from_numpy(G.toarray()).to(device)
del G
gc.collect()

# Add regularization
diag_indices = torch.arange(n_items)
G_dense[diag_indices, diag_indices] += LAMBDA_EASE

# Inversion
print("Matrix inversion...")
start_time = time.time()
P = torch.linalg.inv(G_dense)
print(f"⏱️ Inversion: {time.time() - start_time:.2f}s")

# Cleanup
diag_P = torch.diag(P)
inv_diag_P = 1.0 / diag_P
del G_dense
gc.collect()

# Inference
print("EASE Inference...")
test_users = test_df['user'].unique()
test_user_idxs = [user2idx[u] for u in test_users if u in user2idx]

ease_results = {}
batch_size = 1000

for i in range(0, len(test_user_idxs), batch_size):
    batch_idxs = test_user_idxs[i:i+batch_size]
    x_batch = torch.from_numpy(X[batch_idxs].toarray()).to(device)
    
    # Scores: X - (X @ P) * inv_diag_P
    xp = torch.matmul(x_batch, P)
    scores = x_batch - xp * inv_diag_P.unsqueeze(0)
    scores[x_batch > 0] = -float('inf')
    
    # Top-100
    topk_scores, topk_indices = torch.topk(scores, k=100, dim=1)
    
    for j, u_idx in enumerate(batch_idxs):
        items_np = topk_indices[j].cpu().numpy()
        scores_np = topk_scores[j].cpu().numpy()
        
        # Normalize per user
        min_s, max_s = scores_np.min(), scores_np.max()
        if max_s - min_s > 1e-6:
            norm_scores = (scores_np - min_s) / (max_s - min_s)
        else:
            norm_scores = np.zeros_like(scores_np)
        
        ease_results[u_idx] = {item: score for item, score in zip(items_np, norm_scores)}
    
    if (i // batch_size + 1) % 10 == 0:
        print(f"  Batch {i // batch_size + 1}/{(len(test_user_idxs) + batch_size - 1) // batch_size}")

print(f"✅ EASE complete: {len(ease_results)} users")

# Cleanup
del P, diag_P, inv_diag_P
gc.collect()

#%% [markdown]
# ## Part 3: SimGCL 모델 정의 및 학습

#%% [code]
print("\n" + "="*60)
print("🧠 SimGCL 모델 정의")
print("="*60)

class LightGCN_SimGCL(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, eps=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.eps = eps
        
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
    
    def forward(self, edge_index, edge_weight, perturbed=False):
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        
        if perturbed and self.training:
            random_noise = torch.randn_like(all_emb).to(all_emb.device)
            all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
        
        embs = [all_emb]
        for _ in range(self.n_layers):
            row, col = edge_index
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            all_emb = torch.zeros_like(all_emb).scatter_add(
                0, row.unsqueeze(1).expand(-1, self.emb_dim), messages
            )
            embs.append(all_emb)
        
        final_emb = torch.mean(torch.stack(embs), dim=0)
        return final_emb[:self.n_users], final_emb[self.n_users:]

def compute_infonce_loss(emb_1, emb_2, temperature=0.2):
    emb_1 = F.normalize(emb_1, dim=-1)
    emb_2 = F.normalize(emb_2, dim=-1)
    pos_score = (emb_1 * emb_2).sum(dim=-1) / temperature
    all_scores = torch.matmul(emb_1, emb_2.t()) / temperature
    loss = -torch.log(torch.exp(pos_score) / torch.exp(all_scores).sum(dim=1)).mean()
    return loss

print("✅ SimGCL model defined")

#%% [code]
print("\n" + "="*60)
print("🏋️ SimGCL 학습")
print("="*60)

# Hyperparameters
EMB_DIM = 64
N_LAYERS = 3
EPOCHS = 100
BATCH_SIZE = 2048
LR = 0.001
LAMBDA_CL = 0.2
TEMPERATURE = 0.2
EPS = 0.1

model = LightGCN_SimGCL(n_users, n_items, EMB_DIM, N_LAYERS, EPS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Prepare training data
pos_edges = torch.LongTensor(train_df[['user_idx', 'item_idx']].values).to(device)
n_batches = (len(pos_edges) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Batches/Epoch: {n_batches}")

best_recall = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    
    # Shuffle
    perm = torch.randperm(len(pos_edges))
    pos_edges_shuffled = pos_edges[perm]
    
    for i in range(n_batches):
        batch_pos = pos_edges_shuffled[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        u_idx = batch_pos[:, 0]
        i_idx = batch_pos[:, 1]
        
        # Negative sampling (simplified)
        neg_idx = torch.randint(0, n_items, (len(u_idx),), device=device)
        
        # Forward
        user_emb, item_emb = model(edge_index, edge_weight, perturbed=False)
        
        # BPR Loss
        pos_scores = (user_emb[u_idx] * item_emb[i_idx]).sum(dim=-1)
        neg_scores = (user_emb[u_idx] * item_emb[neg_idx]).sum(dim=-1)
        bpr_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()
        
        # Contrastive Loss
        u_emb_1, i_emb_1 = model(edge_index, edge_weight, perturbed=True)
        u_emb_2, i_emb_2 = model(edge_index, edge_weight, perturbed=True)
        cl_loss = compute_infonce_loss(u_emb_1[u_idx], u_emb_2[u_idx], TEMPERATURE) + \
                  compute_infonce_loss(i_emb_1[i_idx], i_emb_2[i_idx], TEMPERATURE)
        
        # Total Loss
        loss = bpr_loss + LAMBDA_CL * cl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / n_batches
    
    # Validation every 5 epochs
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model(edge_index, edge_weight, perturbed=False)
            
            # Simple Recall@20 on val
            val_users = val_df['user_idx'].unique()
            hits = 0
            total = 0
            
            for u_idx in val_users[:1000]:  # Sample for speed
                seen = user_train_items.get(u_idx, set())
                val_items = set(val_df[val_df['user_idx'] == u_idx]['item_idx'].values)
                
                scores = (user_emb[u_idx] * item_emb).sum(dim=-1)
                scores[list(seen)] = -float('inf')
                
                top20 = torch.topk(scores, k=20)[1].cpu().tolist()
                hits += len(set(top20) & val_items)
                total += len(val_items)
            
            recall = hits / total if total > 0 else 0.0
            
            print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Recall@20: {recall:.4f}")
            
            if recall > best_recall:
                best_recall = recall
                torch.save({
                    'state_dict': model.state_dict(),
                    'config': {'n_users': n_users, 'n_items': n_items, 
                               'emb_dim': EMB_DIM, 'n_layers': N_LAYERS, 'eps': EPS}
                }, f'{MODEL_DIR}/simgcl_best.pt')
                print(f"  💾 Best model saved (Recall@20: {best_recall:.4f})")
    else:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

print(f"✅ SimGCL training complete (Best Recall@20: {best_recall:.4f})")

#%% [markdown]
# ## Part 4: SimGCL 추론

#%% [code]
print("\n" + "="*60)
print("🔮 SimGCL 추론")
print("="*60)

# Load best model
checkpoint = torch.load(f'{MODEL_DIR}/simgcl_best.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

simgcl_results = {}

with torch.no_grad():
    user_emb, item_emb = model(edge_index, edge_weight, perturbed=False)
    
    for i in range(0, len(test_user_idxs), batch_size):
        batch_idxs = test_user_idxs[i:i+batch_size]
        batch_u_emb = user_emb[batch_idxs]
        batch_scores = torch.matmul(batch_u_emb, item_emb.t())
        
        for j, u_idx in enumerate(batch_idxs):
            scores = batch_scores[j]
            seen = user_train_items.get(u_idx, set())
            if seen:
                scores[list(seen)] = -float('inf')
            
            topk_scores, topk_indices = torch.topk(scores, k=100)
            
            items_np = topk_indices.cpu().numpy()
            scores_np = topk_scores.cpu().numpy()
            
            # Normalize
            min_s, max_s = scores_np.min(), scores_np.max()
            if max_s - min_s > 1e-6:
                norm_scores = (scores_np - min_s) / (max_s - min_s)
            else:
                norm_scores = np.zeros_like(scores_np)
            
            simgcl_results[u_idx] = {item: score for item, score in zip(items_np, norm_scores)}

print(f"✅ SimGCL inference complete: {len(simgcl_results)} users")

#%% [markdown]
# ## Part 5: 앙상블

#%% [code]
print("\n" + "="*60)
print("🎯 앙상블 (SimGCL + EASE)")
print("="*60)

ALPHA = 0.5  # SimGCL weight

ensemble_preds = []

for u in test_users:
    if u not in user2idx:
        continue
    
    u_idx = user2idx[u]
    K = user_k.get(u_idx, 0)
    num_recommend = int(K * 0.5)
    
    if num_recommend == 0:
        continue
    
    # Get scores
    simgcl_map = simgcl_results.get(u_idx, {})
    ease_map = ease_results.get(u_idx, {})
    
    # Merge
    all_candidates = set(simgcl_map.keys()) | set(ease_map.keys())
    merged_scores = []
    
    for item in all_candidates:
        s_score = simgcl_map.get(item, 0.0)
        e_score = ease_map.get(item, 0.0)
        final_score = ALPHA * s_score + (1 - ALPHA) * e_score
        merged_scores.append((item, final_score))
    
    # Top-N
    merged_scores.sort(key=lambda x: x[1], reverse=True)
    top_n = merged_scores[:num_recommend]
    rec_items = {idx2item[item] for item, _ in top_n}
    
    ensemble_preds.append((u, rec_items))

# Convert to submission
user_rec_map = {u: recs for u, recs in ensemble_preds}
submission_rows = []

for _, row in test_df.iterrows():
    u, i = row['user'], row['item']
    recs = user_rec_map.get(u, set())
    recommend = 'O' if i in recs else 'X'
    submission_rows.append({'user': u, 'item': i, 'recommend': recommend})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(f'{OUTPUT_DIR}/ensemble_predictions.csv', index=False)

rec_cnt = len(submission_df[submission_df['recommend'] == 'O'])
total_cnt = len(submission_df)

print(f"✅ Ensemble complete")
print(f"📊 Recommendations: {rec_cnt}/{total_cnt} ({rec_cnt/total_cnt*100:.2f}%)")
print(f"💾 Saved to {OUTPUT_DIR}/ensemble_predictions.csv")

#%% [markdown]
# ## Part 6: 평가 (Validation Set)

#%% [code]
print("\n" + "="*60)
print("📈 평가")
print("="*60)

def evaluate_metrics(model, user_emb, item_emb, eval_df, user_train_items, k=20):
    """Recall, NDCG, Precision, MAP 계산"""
    from collections import defaultdict
    
    user_metrics = defaultdict(lambda: {'hits': 0, 'total': 0, 'dcg': 0.0, 'idcg': 0.0, 'ap': 0.0})
    
    for u_idx in eval_df['user_idx'].unique():
        seen = user_train_items.get(u_idx, set())
        true_items = set(eval_df[eval_df['user_idx'] == u_idx]['item_idx'].values)
        
        scores = (user_emb[u_idx] * item_emb).sum(dim=-1)
        scores[list(seen)] = -float('inf')
        
        topk_items = torch.topk(scores, k=k)[1].cpu().tolist()
        
        # Metrics
        hits = len(set(topk_items) & true_items)
        user_metrics[u_idx]['hits'] = hits
        user_metrics[u_idx]['total'] = len(true_items)
        
        # NDCG
        dcg = sum([(1.0 if item in true_items else 0.0) / np.log2(i + 2) for i, item in enumerate(topk_items)])
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_items), k))])
        user_metrics[u_idx]['dcg'] = dcg
        user_metrics[u_idx]['idcg'] = idcg
        
        # MAP
        precisions = []
        hits_so_far = 0
        for i, item in enumerate(topk_items):
            if item in true_items:
                hits_so_far += 1
                precisions.append(hits_so_far / (i + 1))
        user_metrics[u_idx]['ap'] = np.mean(precisions) if precisions else 0.0
    
    # Aggregate
    total_hits = sum([m['hits'] for m in user_metrics.values()])
    total_items = sum([m['total'] for m in user_metrics.values()])
    recall = total_hits / total_items if total_items > 0 else 0.0
    
    ndcg = np.mean([m['dcg'] / m['idcg'] if m['idcg'] > 0 else 0.0 for m in user_metrics.values()])
    precision = total_hits / (len(user_metrics) * k)
    map_score = np.mean([m['ap'] for m in user_metrics.values()])
    
    return {'Recall': recall, 'NDCG': ndcg, 'Precision': precision, 'MAP': map_score}

# Evaluate
model.eval()
with torch.no_grad():
    user_emb, item_emb = model(edge_index, edge_weight, perturbed=False)
    metrics = evaluate_metrics(model, user_emb, item_emb, val_df, user_train_items, k=20)

print(f"📊 Validation Metrics @20:")
print(f"  Recall:    {metrics['Recall']:.4f}")
print(f"  NDCG:      {metrics['NDCG']:.4f}")
print(f"  Precision: {metrics['Precision']:.4f}")
print(f"  MAP:       {metrics['MAP']:.4f}")

print("\n✅ 파이프라인 완료!")
