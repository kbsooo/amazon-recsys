import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import scipy.sparse as sp
import os

# 시드 고정
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# 디바이스 설정
# MPS는 Sparse Tensor 연산(aten::addmm) 지원이 미비하여 CPU 사용
device = torch.device('cpu')
print(f"Using device: {device}")

# 데이터 로드
print("Loading data...")
base_dir = os.path.dirname(os.path.abspath(__file__))
train_df = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))
test_df = pd.read_csv(os.path.join(base_dir, 'test_data.csv'))

with open(os.path.join(base_dir, 'user_mapper.pkl'), 'rb') as f:
    user_mapper = pickle.load(f)
with open(os.path.join(base_dir, 'item_mapper.pkl'), 'rb') as f:
    item_mapper = pickle.load(f)

n_users = len(user_mapper)
n_items = len(item_mapper)

print(f"Users: {n_users}, Items: {n_items}")

# Adjacency Matrix 생성 (Sparse Tensor)
def create_adj_matrix(df, n_users, n_items):
    u = df['user_idx'].values
    i = df['item_idx'].values
    
    user_np = np.array(u)
    item_np = np.array(i)
    
    ratings = np.ones_like(user_np, dtype=np.float32)
    
    n_nodes = n_users + n_items
    
    # Direct COO Construction
    tmp_adj = sp.coo_matrix((ratings, (user_np, item_np + n_users)), shape=(n_nodes, n_nodes))
    adj_mat = tmp_adj + tmp_adj.T
    
    # Normalize
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    
    return torch.sparse_coo_tensor(indices, values, shape)

print("Creating Adjacency Matrix...")
adj_matrix = create_adj_matrix(train_df, n_users, n_items)
adj_matrix = adj_matrix.to(device)
print("Adjacency Matrix created.")

# LightGCN 모델 정의
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, adj_matrix):
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
            
        embs = torch.stack(embs, dim=1)
        final_emb = torch.mean(embs, dim=1)
        
        users_emb, items_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return users_emb, items_emb

# BPR Loss
class BPRLoss(nn.Module):
    def __init__(self, decay=1e-4):
        super(BPRLoss, self).__init__()
        self.decay = decay
        
    def forward(self, users_emb, items_emb, users, pos_items, neg_items, current_user_emb, current_pos_item_emb, current_neg_item_emb):
        u_emb = users_emb[users]
        pos_emb = items_emb[pos_items]
        neg_emb = items_emb[neg_items]
        
        pos_scores = torch.sum(u_emb * pos_emb, dim=1)
        neg_scores = torch.sum(u_emb * neg_emb, dim=1)
        
        loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
        
        reg_loss = (1/2) * (current_user_emb.norm(2).pow(2) + 
                            current_pos_item_emb.norm(2).pow(2) + 
                            current_neg_item_emb.norm(2).pow(2)) / float(len(users))
        
        return loss + self.decay * reg_loss

# Dataset
class TrainDataset(Dataset):
    def __init__(self, df, n_items):
        self.users = df['user_idx'].values
        self.items = df['item_idx'].values
        self.n_items = n_items
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user = self.users[idx]
        pos_item = self.items[idx]
        
        while True:
            neg_item = np.random.randint(0, self.n_items)
            if neg_item != pos_item:
                break
                
        return user, pos_item, neg_item

# Hyperparameters
EMB_DIM = 64
N_LAYERS = 2
BATCH_SIZE = 2048
LR = 0.001
EPOCHS = 5
DECAY = 1e-4

print("Initializing Model...")
model = LightGCN(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = BPRLoss(decay=DECAY)

train_dataset = TrainDataset(train_df, n_items)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Starting Training...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for users, pos_items, neg_items in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        
        optimizer.zero_grad()
        
        users_emb, items_emb = model(adj_matrix)
        
        current_user_emb = model.user_embedding(users)
        current_pos_item_emb = model.item_embedding(pos_items)
        current_neg_item_emb = model.item_embedding(neg_items)
        
        loss = criterion(users_emb, items_emb, users, pos_items, neg_items, 
                         current_user_emb, current_pos_item_emb, current_neg_item_emb)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# Save Model
torch.save(model.state_dict(), os.path.join(base_dir, 'lightgcn_model.pt'))
print("Model saved.")

# Evaluation (Custom Logic)
print("Evaluating...")
model.eval()
with torch.no_grad():
    users_emb, items_emb = model(adj_matrix)

user_history = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

def get_recommendations(user_idx, k=20):
    u_emb = users_emb[user_idx]
    scores = torch.matmul(items_emb, u_emb)
    
    seen_items = user_history.get(user_idx, [])
    scores[seen_items] = -float('inf')
    
    _, top_k_items = torch.topk(scores, k)
    return top_k_items.cpu().numpy()

results = []
test_users = test_df['user_idx'].unique()

# Sample 100 users for quick check
sample_users = np.random.choice(test_users, 100, replace=False)

for u_idx in tqdm(sample_users, desc="Generating Recommendations"):
    history_count = len(user_history.get(u_idx, []))
    
    if history_count <= 10:
        num_to_recommend = 2
    else:
        num_to_recommend = int(history_count * 0.5)
        if num_to_recommend < 1: num_to_recommend = 1
    
    recs = get_recommendations(u_idx, k=num_to_recommend)
    
    gt_items = test_df[test_df['user_idx'] == u_idx]['item_idx'].values
    
    hits = np.intersect1d(recs, gt_items)
    recall = len(hits) / len(gt_items) if len(gt_items) > 0 else 0
    
    results.append({
        'user_idx': u_idx,
        'history_count': history_count,
        'num_recommends': num_to_recommend,
        'hits': len(hits),
        'recall': recall
    })

results_df = pd.DataFrame(results)
print("Average Recall (Sampled):", results_df['recall'].mean())
print(results_df.head())
