#%%
"""
SimGCL 학습 파이프라인 - Amazon RecSys GNN
LightGCN + SimGCL (Contrastive Learning) + Weighted BPR Loss

Kaggle/Colab 업로드용 버전 (Mini - Fast Training)
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
import os

# Create directories for outputs
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore')

# Font settings removed to avoid font errors in Kaggle

#%% [markdown]
# Model Definitions: LightGCN + SimGCL
# ==========================================

#%% [code]

class LightGCN_SimGCL(nn.Module):
    """
    LightGCN with Simple Graph Contrastive Learning (SimGCL)
    
    Uses Contrastive Learning to learn robust embeddings on sparse data.
    Adds noise to embeddings during training for data augmentation effect.
    """
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3, eps=0.1):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        self.eps = eps  # Noise level for perturbation
        
        # Embedding Layers
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
    
    def forward(self, edge_index, edge_weight, perturbed=False):
        """
        Perform Graph Convolution
        
        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            perturbed: If True, return noisy embeddings
        
        Returns:
            user_emb: [n_users, emb_dim]
            item_emb: [n_items, emb_dim]
        """
        # Initial embeddings
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        
        # Perturbation (add noise)
        if perturbed and self.training:
            random_noise = torch.randn_like(all_emb).to(all_emb.device)
            all_emb = all_emb + torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
        
        embs = [all_emb]
        
        # Layer-wise Propagation
        for _ in range(self.n_layers):
            row, col = edge_index
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            all_emb = torch.zeros_like(all_emb).scatter_add(
                0, row.unsqueeze(1).expand(-1, self.emb_dim), messages
            )
            embs.append(all_emb)
        
        # Layer Combination
        final_emb = torch.mean(torch.stack(embs), dim=0)
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def get_perturbed_embeddings(self, edge_index, edge_weight):
        """
        Generate two perturbed views for Contrastive Learning
        """
        u_emb_1, i_emb_1 = self.forward(edge_index, edge_weight, perturbed=True)
        u_emb_2, i_emb_2 = self.forward(edge_index, edge_weight, perturbed=True)
        return u_emb_1, i_emb_1, u_emb_2, i_emb_2
    
    def predict(self, user_idx, item_idx, edge_index, edge_weight):
        """Prediction score (for inference, perturbed=False)"""
        user_emb, item_emb = self.forward(edge_index, edge_weight, perturbed=False)
        scores = (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)
        return scores

def compute_infonce_loss(emb_1, emb_2, temperature=0.2):
    """
    InfoNCE Loss for Contrastive Learning
    
    Args:
        emb_1: [batch_size, emb_dim] - First view
        emb_2: [batch_size, emb_dim] - Second view
        temperature: Temperature parameter
    
    Returns:
        loss: Scalar contrastive loss
    """
    # Normalize embeddings
    emb_1 = F.normalize(emb_1, dim=-1)
    emb_2 = F.normalize(emb_2, dim=-1)
    
    batch_size = emb_1.shape[0]
    
    # Positive pairs: (emb_1[i], emb_2[i])
    pos_score = (emb_1 * emb_2).sum(dim=-1) / temperature  # [batch_size]
    
    # All pairs: emb_1 @ emb_2.T
    all_scores = torch.matmul(emb_1, emb_2.t()) / temperature  # [batch_size, batch_size]
    
    # InfoNCE: -log(exp(pos) / sum(exp(all)))
    loss = -torch.log(
        torch.exp(pos_score) / torch.exp(all_scores).sum(dim=1)
    ).mean()
    
    return loss

print("✅ Model definitions complete")

#%% [markdown]
# 1. Environment Setup
# ==========================================

#%% [code]

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

# Hyperparameters (Mini Version)
EMB_DIM = 32        # Reduced from 64
N_LAYERS = 2        # Reduced from 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 20         # Reduced from 100
BATCH_SIZE = 4096   # Increased from 2048
NUM_NEG = 2         # Reduced from 4
HARD_NEG_RATIO = 0.5

# SimGCL 전용 하이퍼파라미터
EPS = 0.1  # Noise level for perturbation
LAMBDA_CL = 0.2  # Contrastive loss weight
TEMPERATURE = 0.2  # Temperature for InfoNCE

print(f"\n하이퍼파라미터:")
print(f"  임베딩 차원: {EMB_DIM}")
print(f"  레이어 수: {N_LAYERS}")
print(f"  학습률: {LR}")
print(f"  배치 크기: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Noise (eps): {EPS}")
print(f"  Lambda CL: {LAMBDA_CL}")

#%% [markdown]
# 2. Data Loading
# ==========================================

#%% [code]

# Load split data
train_df = pd.read_csv('/kaggle/input/amazon/train_split.csv')
val_df = pd.read_csv('/kaggle/input/amazon/val_split.csv')
test_df = pd.read_csv('/kaggle/input/amazon/test_split.csv')

print(f"Train: {len(train_df):,}")
print(f"Val: {len(val_df):,}")
print(f"Test: {len(test_df):,}")

# Load graph data
graph_data = torch.load('/kaggle/input/amazon/train_graph.pt')
edge_index = graph_data['edge_index'].to(device)
cca_edge_weight = graph_data['cca_weight'].to(device)
n_users = graph_data['n_users']
n_items = graph_data['n_items']

print(f"\n그래프 정보:")
print(f"  유저 수: {n_users:,}")
print(f"  아이템 수: {n_items:,}")
print(f"  엣지 수: {edge_index.shape[1]:,}")

# Load ID mappings for inference
with open('/kaggle/input/amazon/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)
with open('/kaggle/input/amazon/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)
with open('/kaggle/input/amazon/user_k.pkl', 'rb') as f:
    user_k = pickle.load(f)
with open('/kaggle/input/amazon/user_train_items.pkl', 'rb') as f:
    user_train_items = pickle.load(f)

print(f"✅ ID mappings loaded for inference")

#%% [markdown]
# 3. Negative Sampling Functions
# ==========================================

#%% [code]

def fast_sample_negatives(batch_size, num_neg=4):
    """빠른 랜덤 negative sampling"""
    return torch.randint(0, n_items, (batch_size, num_neg), device=device)

@torch.no_grad()
def hard_negative_sampling(user_emb, item_emb, pos_users, num_neg=4, num_candidates=50):
    """Hard Negative Sampling"""
    batch_size = len(pos_users)
    candidates = torch.randint(0, n_items, (batch_size, num_candidates), device=device)
    
    user_expanded = user_emb[pos_users].unsqueeze(1)
    item_candidates = item_emb[candidates]
    scores = (user_expanded * item_candidates).sum(dim=2)
    
    _, top_indices = scores.topk(num_neg, dim=1)
    hard_negs = candidates.gather(1, top_indices)
    
    return hard_negs

print("✅ Negative Sampling 함수 정의 완료")

#%% [markdown]
# 4. Evaluation Functions
# ==========================================

#%% [code]

@torch.no_grad()
def evaluate_recall_ndcg(model, eval_df, edge_index, edge_weight, k_list=[20, 50]):
    """Recall@K 및 NDCG@K 평가"""
    model.eval()
    u_emb, i_emb = model(edge_index, edge_weight, perturbed=False)
    
    user_groups = eval_df.groupby('user_idx')
    
    recall_at_k = {k: [] for k in k_list}
    ndcg_at_k = {k: [] for k in k_list}
    
    for user_idx, group in user_groups:
        gt_items = set(group['item_idx'].values)
        
        user_vec = u_emb[user_idx]
        scores = (user_vec @ i_emb.t()).cpu().numpy()
        
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
    
    metrics = {}
    for k in k_list:
        metrics[f'Recall@{k}'] = np.mean(recall_at_k[k])
        metrics[f'NDCG@{k}'] = np.mean(ndcg_at_k[k])
    
    return metrics

print("✅ 평가 함수 정의 완료")

#%% [markdown]
# 5. SimGCL Training Loop
# ==========================================

#%% [code]

# 모델 초기화
model = LightGCN_SimGCL(n_users, n_items, EMB_DIM, N_LAYERS, eps=EPS).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 학습 데이터
train_users = torch.LongTensor(train_df['user_idx'].values).to(device)
train_items = torch.LongTensor(train_df['item_idx'].values).to(device)
train_ratings = torch.FloatTensor(train_df['rating'].values).to(device)

# [핵심] Rating을 가중치로 변환 (1점=0.5x, 3점=1.0x, 5점=2.0x)
# Formula: weight = 0.3 + 0.34 * rating (1점: 0.64, 3점: 1.32, 5점: 2.0)
train_weights = 0.3 + 0.34 * train_ratings
train_weights = train_weights.to(device)

# 학습 이력
history = {
    'loss': [], 
    'bpr_loss': [], 
    'cl_loss': [],
    'val_recall@20': [], 
    'val_ndcg@20': []
}

print("학습 시작...")
best_val_recall = 0
patience_counter = 0
PATIENCE = 15

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(train_users), device=device)
    epoch_loss = 0
    epoch_bpr = 0
    epoch_cl = 0
    n_batches = 0
    
    for i in range(0, len(train_users), BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        pos_u = train_users[batch_idx]
        pos_i = train_items[batch_idx]
        weights = train_weights[batch_idx]
        
        # ========== Weighted BPR Loss ==========
        u_emb, i_emb = model(edge_index, cca_edge_weight, perturbed=False)
        
        # Hard + Random Negative Sampling
        n_hard = int(NUM_NEG * HARD_NEG_RATIO)
        hard_negs = hard_negative_sampling(u_emb, i_emb, pos_u, num_neg=n_hard)
        rand_negs = fast_sample_negatives(len(batch_idx), NUM_NEG - n_hard)
        neg_i = torch.cat([hard_negs, rand_negs], dim=1)
        
        # BPR Loss with Rating Weights
        pos_scores = (u_emb[pos_u] * i_emb[pos_i]).sum(dim=1)
        neg_scores = (u_emb[pos_u].unsqueeze(1) * i_emb[neg_i]).sum(dim=2)
        diff = pos_scores.unsqueeze(1) - neg_scores
        loss_bpr_per_sample = -torch.log(torch.sigmoid(diff) + 1e-8).mean(dim=1)
        loss_bpr = (loss_bpr_per_sample * weights).mean()
        
        # ========== Contrastive Loss (InfoNCE) ==========
        # Perturbed views 생성
        u_emb_1, i_emb_1, u_emb_2, i_emb_2 = model.get_perturbed_embeddings(edge_index, cca_edge_weight)
        
        # User Contrastive Loss (배치 내에서)
        u_cl = compute_infonce_loss(u_emb_1[pos_u], u_emb_2[pos_u], temperature=TEMPERATURE)
        
        # Item Contrastive Loss (배치 내에서)
        i_cl = compute_infonce_loss(i_emb_1[pos_i], i_emb_2[pos_i], temperature=TEMPERATURE)
        
        loss_cl = (u_cl + i_cl) / 2
        
        # ========== Total Loss ==========
        total_loss = loss_bpr + LAMBDA_CL * loss_cl
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
        epoch_bpr += loss_bpr.item()
        epoch_cl += loss_cl.item()
        n_batches += 1
    
    scheduler.step()
    avg_loss = epoch_loss / n_batches
    avg_bpr = epoch_bpr / n_batches
    avg_cl = epoch_cl / n_batches
    
    history['loss'].append(avg_loss)
    history['bpr_loss'].append(avg_bpr)
    history['cl_loss'].append(avg_cl)
    
    # Validation (매 5 epoch)
    if (epoch + 1) % 5 == 0:
        val_metrics = evaluate_recall_ndcg(model, val_df, edge_index, cca_edge_weight, k_list=[20])
        history['val_recall@20'].append(val_metrics['Recall@20'])
        history['val_ndcg@20'].append(val_metrics['NDCG@20'])
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
              f"BPR: {avg_bpr:.4f} | CL: {avg_cl:.4f} | "
              f"Val Recall@20: {val_metrics['Recall@20']:.4f} | "
              f"Val NDCG@20: {val_metrics['NDCG@20']:.4f}")
        
        # Early Stopping
        if val_metrics['Recall@20'] > best_val_recall:
            best_val_recall = val_metrics['Recall@20']
            patience_counter = 0
            # Save model with config
            torch.save({
                'state_dict': model.state_dict(),
                'config': {
                    'n_users': n_users,
                    'n_items': n_items,
                    'emb_dim': EMB_DIM,
                    'n_layers': N_LAYERS,
                    'eps': EPS
                }
            }, 'models/simgcl_mini_best.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f"\n✅ SimGCL 모델 학습 완료 (Best Val Recall@20: {best_val_recall:.4f})")

#%% [markdown]
# 6. Training Curves Visualization
# ==========================================

#%% [code]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Total Loss
axes[0].plot(history['loss'], label='Total Loss', color='blue')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Total Training Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# BPR vs CL Loss
axes[1].plot(history['bpr_loss'], label='BPR Loss', color='orange')
axes[1].plot(history['cl_loss'], label='CL Loss', color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('BPR vs Contrastive Loss')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Validation Metrics
if history['val_recall@20']:
    x = np.arange(5, len(history['loss']) + 1, 5)[:len(history['val_recall@20'])]
    axes[2].plot(x, history['val_recall@20'], label='Recall@20', marker='o', color='purple')
    axes[2].plot(x, history['val_ndcg@20'], label='NDCG@20', marker='s', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score')
    axes[2].set_title('Validation Metrics')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.tight_layout()
plt.savefig('outputs/simgcl_mini_training_curves.png', dpi=150)
plt.show()

print("✅ Training curves saved (outputs/simgcl_mini_training_curves.png)")

#%% [markdown]
# 7. Save Final Model
# ==========================================

#%% [code]

# Load best model (Best validation)
checkpoint = torch.load('models/simgcl_mini_best.pt')
model.load_state_dict(checkpoint['state_dict'])

# Save complete checkpoint
torch.save({
    'state_dict': model.state_dict(),
    'config': {
        'n_users': n_users,
        'n_items': n_items,
        'emb_dim': EMB_DIM,
        'n_layers': N_LAYERS,
        'eps': EPS
    },
    'history': history
}, 'models/simgcl_mini_final.pt')

print("✅ Model saved:")
print("  - models/simgcl_mini_best.pt")
print("  - models/simgcl_mini_final.pt")

print("\n✅ SimGCL 학습 파이프라인 실행 완료!")

#%% [markdown]
# 8. Inference
# ==========================================

#%% [code]
class SimGCLInference:
    def __init__(self, model_path, data_dir, device):
        self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        config = checkpoint['config']
        
        # Load ID mappings
        with open(f'{data_dir}/user2idx.pkl', 'rb') as f:
            self.user2idx = pickle.load(f)
        with open(f'{data_dir}/item2idx.pkl', 'rb') as f:
            self.item2idx = pickle.load(f)
        with open(f'{data_dir}/user_k.pkl', 'rb') as f:
            self.user_k = pickle.load(f)
        with open(f'{data_dir}/user_train_items.pkl', 'rb') as f:
            self.user_train_items = pickle.load(f)
            
        # Load graph data
        graph_data = torch.load(f'{data_dir}/train_graph.pt', map_location=self.device)
        self.edge_index = graph_data['edge_index'].to(self.device)
        self.edge_weight = graph_data['cca_weight'].to(self.device)
        
        # Initialize and load model
        self.model = LightGCN_SimGCL(
            config['n_users'],
            config['n_items'],
            config['emb_dim'],
            config['n_layers'],
            eps=config['eps']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        print("✅ Inference system initialized")
        
        # Pre-compute embeddings
        with torch.no_grad():
            self.user_emb, self.item_emb = self.model(self.edge_index, self.edge_weight, perturbed=False)
    
    def predict(self, test_df):
        results = []
        
        # Group by user for batch processing
        for user_id, group in test_df.groupby('user'):
            if user_id not in self.user2idx:
                for _, row in group.iterrows():
                    results.append({'user': row['user'], 'item': row['item'], 'recommend': 'X'})
                continue
            
            u_idx = self.user2idx[user_id]
            K = self.user_k.get(u_idx, 2)
            MIN_K = 2
            
            items_to_score = []
            
            for _, row in group.iterrows():
                item_id = row['item']
                if item_id not in self.item2idx:
                    results.append({'user': row['user'], 'item': row['item'], 'recommend': 'X'})
                    continue
                
                i_idx = self.item2idx[item_id]
                # Filter seen items
                if i_idx in self.user_train_items.get(u_idx, set()):
                    results.append({'user': row['user'], 'item': row['item'], 'recommend': 'X'})
                    continue
                
                items_to_score.append((i_idx, row))
            
            if not items_to_score:
                continue
            
            # Calculate scores
            item_indices = torch.LongTensor([i for i, _ in items_to_score]).to(self.device)
            
            with torch.no_grad():
                scores = (self.user_emb[u_idx] * self.item_emb[item_indices]).sum(dim=1).cpu().numpy()
            
            # Top-K Selection (50% Rule)
            num_recommend = max(MIN_K, min(K, len(scores) // 2))
            top_indices = np.argsort(scores)[-num_recommend:]
            top_set = set(top_indices)
            
            for idx, (item_idx, row) in enumerate(items_to_score):
                recommend = 'O' if idx in top_set else 'X'
                results.append({'user': row['user'], 'item': row['item'], 'recommend': recommend})
        
        return pd.DataFrame(results)

# Run Inference
print("\n" + "="*60)
print("Running Inference on Test Set")
print("="*60)

# Initialize Inference
inference = SimGCLInference(
    model_path='models/simgcl_mini_final.pt',
    data_dir='/kaggle/input/amazon',
    device=device
)

# Run prediction on test set
print("Generating predictions...")
predictions = inference.predict(test_df[['user', 'item']])

# Save results
output_path = 'outputs/predictions.csv'
predictions.to_csv(output_path, index=False)
print(f"✅ Predictions saved to {output_path}")

# Show sample
print("\nSample Predictions:")
print(predictions.head(10))

# Stats
rec_cnt = len(predictions[predictions['recommend'] == 'O'])
total_cnt = len(predictions)
print(f"\nTotal Recommendations: {rec_cnt}/{total_cnt} ({rec_cnt/total_cnt*100:.2f}%)")
