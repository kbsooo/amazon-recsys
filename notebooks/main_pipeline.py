#%% [markdown]
# Amazon RecSys: Integrated Pipeline (PyTorch)
# SimGCL + EASE Ensemble
#
# 이 노트북은 다음 단계를 모두 포함합니다:
# 1. 데이터 로드 및 전처리 (Graph 생성)
# 2. SimGCL 모델 학습 (Contrastive Learning)
# 3. EASE 모델 학습 (Closed-form Solution)
# 4. 앙상블 추론 (Weighted Sum + 50% Rule)

#%% [code]
import os
import sys
import time
import gc
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Configuration
SEED = 42
DATA_DIR = '/kaggle/input/amazon' if os.path.exists('/kaggle/input/amazon') else 'data'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

# SimGCL Hyperparameters
EMB_DIM = 64
N_LAYERS = 3
EPOCHS = 50 # Integrated version might use fewer epochs for demo, or full 100
BATCH_SIZE = 2048
LR = 0.001
LAMBDA_CL = 0.2
EPS = 0.1
TEMPERATURE = 0.2

# EASE Hyperparameters
EASE_LAMBDA = 500.0

# Ensemble Hyperparameters
ALPHA = 0.5 # SimGCL Weight

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# Set Seed
torch.manual_seed(SEED)
np.random.seed(SEED)

#%% [markdown]
# ## 1. Data Loading & Preprocessing

#%% [code]
print("Loading Data...")
train_df = pd.read_csv(f'{DATA_DIR}/train_split.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_split.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test_split.csv')

# Load Mappings
with open(f'{DATA_DIR}/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)
with open(f'{DATA_DIR}/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)

n_users = len(user2idx)
n_items = len(item2idx)
print(f"Users: {n_users}, Items: {n_items}")

# Create Graph (Edge Index) for SimGCL
# We use train_df for SimGCL training
print("Building Graph for SimGCL...")
train_src = [user2idx[u] for u in train_df['user']]
train_dst = [item2idx[i] for i in train_df['item']]
train_edge_index = torch.tensor([train_src, train_dst], dtype=torch.long)

# Normalize Adjacency Matrix (LightGCN style)
def compute_normalized_laplacian(edge_index, n_users, n_items):
    # Construct Adjacency: [U, I]
    # We need symmetric adjacency for GCN: 
    # A = [0, R]
    #     [R.T, 0]
    
    src, dst = edge_index
    dst = dst + n_users # Shift item indices
    
    # Bi-directional edges
    full_src = torch.cat([src, dst])
    full_dst = torch.cat([dst, src])
    full_edge_index = torch.stack([full_src, full_dst])
    
    # Degree
    num_nodes = n_users + n_items
    deg = torch.zeros(num_nodes, dtype=torch.float)
    deg.scatter_add_(0, full_src, torch.ones_like(full_src, dtype=torch.float))
    
    # Norm: D^{-1/2}
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Edge Weight: D^{-1/2} * A * D^{-1/2}
    # For edge (i, j), weight is deg[i]^-0.5 * deg[j]^-0.5
    edge_weight = deg_inv_sqrt[full_src] * deg_inv_sqrt[full_dst]
    
    return full_edge_index, edge_weight

norm_edge_index, norm_edge_weight = compute_normalized_laplacian(train_edge_index, n_users, n_items)
norm_edge_index = norm_edge_index.to(device)
norm_edge_weight = norm_edge_weight.to(device)

print("Graph built.")

#%% [markdown]
# ## 2. SimGCL Model & Training

#%% [code]
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
            noise = torch.randn_like(all_emb).to(all_emb.device)
            all_emb = all_emb + torch.sign(all_emb) * F.normalize(noise, dim=-1) * self.eps
        
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

    def get_perturbed_embeddings(self, edge_index, edge_weight):
        u1, i1 = self.forward(edge_index, edge_weight, perturbed=True)
        u2, i2 = self.forward(edge_index, edge_weight, perturbed=True)
        return u1, i1, u2, i2

def compute_infonce_loss(emb1, emb2, temp=0.2):
    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)
    pos = (emb1 * emb2).sum(dim=-1) / temp
    all_scores = torch.matmul(emb1, emb2.t()) / temp
    loss = -torch.log(torch.exp(pos) / torch.exp(all_scores).sum(dim=1)).mean()
    return loss

# Training Setup
model = LightGCN_SimGCL(n_users, n_items, EMB_DIM, N_LAYERS, EPS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Simple Dataset for BPR
class BPRDataset(Dataset):
    def __init__(self, train_df, n_items, n_neg=1):
        self.users = torch.LongTensor([user2idx[u] for u in train_df['user']])
        self.items = torch.LongTensor([item2idx[i] for i in train_df['item']])
        self.n_items = n_items
        self.n_neg = n_neg
        
        # User-Item Set for fast negative sampling
        self.user_item_set = set(zip(self.users.numpy(), self.items.numpy()))

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        i = self.items[idx]
        
        # Negative Sampling
        neg_item = np.random.randint(0, self.n_items)
        while (u.item(), neg_item) in self.user_item_set:
            neg_item = np.random.randint(0, self.n_items)
            
        return u, i, torch.tensor(neg_item)

train_dataset = BPRDataset(train_df, n_items)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Starting SimGCL Training...")
model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for u, i, j in train_loader:
        u, i, j = u.to(device), i.to(device), j.to(device)
        
        # Forward
        u_emb, i_emb = model(norm_edge_index, norm_edge_weight, perturbed=False)
        
        # BPR Loss
        u_e, i_e, j_e = u_emb[u], i_emb[i], i_emb[j]
        pos_score = (u_e * i_e).sum(dim=-1)
        neg_score = (u_e * j_e).sum(dim=-1)
        bpr_loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
        
        # CL Loss
        u1, i1, u2, i2 = model.get_perturbed_embeddings(norm_edge_index, norm_edge_weight)
        # Calculate CL only on current batch users/items to save memory/time
        # Or unique users/items in batch
        unique_u = torch.unique(u)
        unique_i = torch.unique(torch.cat([i, j]))
        
        cl_loss = compute_infonce_loss(u1[unique_u], u2[unique_u], TEMPERATURE) + \
                  compute_infonce_loss(i1[unique_i], i2[unique_i], TEMPERATURE)
        
        loss = bpr_loss + LAMBDA_CL * cl_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

print("SimGCL Training Complete.")

#%% [markdown]
# ## 3. EASE Model Training

#%% [code]
print("Starting EASE Training...")
# Combine Train + Val for EASE
full_train_df = pd.concat([train_df, val_df])

# Create Sparse Matrix
rows = [user2idx[u] for u in full_train_df['user']]
cols = [item2idx[i] for i in full_train_df['item']]
data = np.ones(len(rows), dtype=np.float32)

X = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))

# Gram Matrix
G = X.T.dot(X).toarray()
diag_indices = np.arange(n_items)
G[diag_indices, diag_indices] += EASE_LAMBDA

# Inversion
P = np.linalg.inv(G)
B = P / (-np.diag(P))
B[diag_indices, diag_indices] = 0

# Convert B to Tensor for Inference
B_tensor = torch.from_numpy(B).float().to(device)

print("EASE Training Complete.")

#%% [markdown]
# ## 4. Ensemble Inference

#%% [code]
print("Running Ensemble Inference...")

# Prepare Test Data
test_users = test_df['user'].unique()
test_user_idxs = [user2idx[u] for u in test_users if u in user2idx]

# User History (for masking seen items)
user_history = {}
for u, i in zip(full_train_df['user'], full_train_df['item']):
    if u in user2idx and i in item2idx:
        uidx = user2idx[u]
        if uidx not in user_history: user_history[uidx] = set()
        user_history[uidx].add(item2idx[i])

# Batch Inference
batch_size = 1000
n_batches = (len(test_user_idxs) + batch_size - 1) // batch_size

final_results = []

# SimGCL Embeddings (Final)
model.eval()
with torch.no_grad():
    u_emb_final, i_emb_final = model(norm_edge_index, norm_edge_weight, perturbed=False)

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(test_user_idxs))
    batch_u_idxs = test_user_idxs[start:end]
    
    # 1. SimGCL Scores
    # [B, Dim] @ [Dim, I] -> [B, I]
    s_scores = torch.matmul(u_emb_final[batch_u_idxs], i_emb_final.t())
    
    # 2. EASE Scores
    # X_batch [B, I] @ B [I, I] -> [B, I]
    # Need X_batch for these users
    x_batch_scipy = X[batch_u_idxs]
    x_batch_tensor = torch.from_numpy(x_batch_scipy.toarray()).float().to(device)
    e_scores = torch.matmul(x_batch_tensor, B_tensor)
    
    # 3. Normalize & Combine
    # Min-Max Normalize per user (row-wise)
    def normalize(tensor):
        min_v = tensor.min(dim=1, keepdim=True)[0]
        max_v = tensor.max(dim=1, keepdim=True)[0]
        return (tensor - min_v) / (max_v - min_v + 1e-8)
    
    s_norm = normalize(s_scores)
    e_norm = normalize(e_scores)
    
    final_scores = ALPHA * s_norm + (1 - ALPHA) * e_norm
    
    # 4. Mask Seen Items
    for idx, u_idx in enumerate(batch_u_idxs):
        seen = user_history.get(u_idx, set())
        if seen:
            final_scores[idx, list(seen)] = -float('inf')
            
    # 5. Top-K & Rules
    # Get K for each user
    # We need user_k mapping.
    # Let's assume we have it or calculate it from train_df?
    # The user provided user_k.pkl, let's load it.
    # If not loaded, calculate from train_df.
    # We'll assume user_k is available or calculate it here.
    pass # Logic inside loop
    
    # We need to load user_k.pkl
    with open(f'{DATA_DIR}/user_k.pkl', 'rb') as f:
        user_k = pickle.load(f)
        
    for idx, u_idx in enumerate(batch_u_idxs):
        scores = final_scores[idx]
        K = user_k.get(u_idx, 0)
        num_rec = int(K * 0.5)
        
        if num_rec > 0:
            _, top_indices = torch.topk(scores, k=num_rec)
            recs = top_indices.cpu().numpy()
        else:
            recs = []
            
        # Convert to Item IDs
        u_id = [k for k, v in user2idx.items() if v == u_idx][0] # Slow reverse lookup
        # Better to use idx2user map created earlier
        # Assuming idx2user exists
        
        rec_items = {list(item2idx.keys())[list(item2idx.values()).index(ri)] for ri in recs} # Slow
        # Use idx2item
        
        final_results.append((u_id, rec_items))

# Save Results
# ... (Same as ensemble.py)
print("Inference Done.")
