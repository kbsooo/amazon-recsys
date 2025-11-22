#%%
"""
CCB (Rating Prediction) 모델 학습 스크립트
목적: CCA+CCB 앙상블 성능 비교를 위해 CCB 모델을 학습시킴.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
import sys
from pathlib import Path

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from model import LightGCN_Rating

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

# 데이터 경로 설정
data_dir = Path(__file__).parent.parent / 'data'
models_dir = Path(__file__).parent.parent / 'models'
models_dir.mkdir(exist_ok=True)

# 데이터 로드
train_df = pd.read_csv(data_dir / 'train_split.csv')
val_df = pd.read_csv(data_dir / 'val_split.csv')

# 그래프 데이터 로드
graph_data = torch.load(data_dir / 'train_graph.pt')
edge_index = graph_data['edge_index'].to(device)
ccb_edge_weight = graph_data['ccb_weight'].to(device) # CCB용 가중치 사용
n_users = graph_data['n_users']
n_items = graph_data['n_items']

# 학습 설정
EMB_DIM = 64
N_LAYERS = 3
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 5
BATCH_SIZE = 2048
NUM_NEG = 4

# 모델 초기화
model = LightGCN_Rating(n_users, n_items, EMB_DIM, N_LAYERS).to(device)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
mse_loss_fn = nn.MSELoss()

# 학습 데이터 텐서 변환
train_users = torch.LongTensor(train_df['user_idx'].values).to(device)
train_items = torch.LongTensor(train_df['item_idx'].values).to(device)
train_ratings = torch.FloatTensor(train_df['rating'].values).to(device)

print("CCB 모델 학습 시작...")
best_val_rmse = float('inf')
patience = 0
PATIENCE_LIMIT = 5

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(train_users), device=device)
    epoch_loss = 0
    
    for i in range(0, len(train_users), BATCH_SIZE):
        batch_idx = perm[i:i+BATCH_SIZE]
        pos_u = train_users[batch_idx]
        pos_i = train_items[batch_idx]
        pos_r = train_ratings[batch_idx]
        
        # Forward
        pred_rating = model.predict_rating(pos_u, pos_i, edge_index, ccb_edge_weight)
        loss = mse_loss_fn(pred_rating, pos_r)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / (len(train_users) // BATCH_SIZE)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_u = torch.LongTensor(val_df['user_idx'].values).to(device)
        val_i = torch.LongTensor(val_df['item_idx'].values).to(device)
        val_r = torch.FloatTensor(val_df['rating'].values).to(device)
        
        val_pred = model.predict_rating(val_u, val_i, edge_index, ccb_edge_weight)
        val_rmse = torch.sqrt(mse_loss_fn(val_pred, val_r)).item()
        
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Val RMSE: {val_rmse:.4f}")
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        patience = 0
        torch.save(model.state_dict(), models_dir / 'ccb_best.pt')
    else:
        patience += 1
        if patience >= PATIENCE_LIMIT:
            print("Early Stopping")
            break

print(f"CCB 학습 완료. Best Val RMSE: {best_val_rmse:.4f}")
