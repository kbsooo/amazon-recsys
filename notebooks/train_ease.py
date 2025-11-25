#%% [markdown]
# EASE (Embarrassingly Shallow Autoencoders) Implementation
# for Amazon RecSys

#%% [code]
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import pickle
import os
import gc
import time

# Configuration
SEED = 42
LAMBDA = 500.0 # Regularization parameter (Commonly 100~1000)
DATA_DIR = '/kaggle/input/amazon' if os.path.exists('/kaggle/input/amazon') else 'data'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Device
device = torch.device('cpu') # Matrix inversion of 74k x 74k is huge. CPU with swap might be safer than GPU OOM.
if torch.backends.mps.is_available():
    # MPS might not support large matrix inversion or float64. Let's stick to CPU for stability with large RAM/Swap.
    # Or try MPS if user wants speed, but risk OOM.
    print("MPS available but using CPU for large matrix stability.")
    # device = torch.device('mps') 

print(f"Device: {device}")

#%% [code]
# 1. Load Data
print("Loading data...")
train_df = pd.read_csv(f'{DATA_DIR}/train_split.csv')
# We need all interactions for EASE training (train + val) to capture full co-occurrence
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

# Combine train and val for training
full_train_df = pd.concat([train_df, val_df])

#%% [code]
# 2. Create Interaction Matrix X
print("Creating interaction matrix...")
# Rows: Users, Cols: Items
# EASE ignores ratings, uses binary (1) or ratings as weights. Usually binary 1 is fine.
# Let's use binary 1.
rows = full_train_df['user_idx'].values
cols = full_train_df['item_idx'].values
data = np.ones(len(rows), dtype=np.float32)

X = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
print(f"Interaction Matrix Shape: {X.shape}")

#%% [code]
# 3. Compute Gram Matrix G = X.T @ X
print("Computing Gram Matrix G = X.T @ X ...")
# This results in Item x Item matrix
# 74k x 74k float32 = ~21 GB. This is the bottleneck.
# We will try to do this. If it fails, we need a machine with more RAM.

start_time = time.time()
G = X.T.dot(X) # Sparse dot product
print(f"Gram Matrix computed in {time.time() - start_time:.2f}s")

# Convert to dense for inversion
# We might need to convert to dense block by block or just go for it.
# Let's try converting to dense PyTorch tensor.
print("Converting to dense Tensor...")
try:
    G_dense = torch.from_numpy(G.toarray()).to(device)
except Exception as e:
    print(f"Error converting to dense: {e}")
    print("Attempting to use float16 to save memory...")
    G_dense = torch.from_numpy(G.toarray().astype(np.float16)).to(device).float() # Convert back to float32 for inversion

# Free sparse G
del G
gc.collect()

#%% [code]
# 4. EASE Training (Inversion)
print("EASE Training (Matrix Inversion)...")
# G += lambda * I
diag_indices = torch.arange(n_items)
G_dense[diag_indices, diag_indices] += LAMBDA

# P = inv(G)
start_time = time.time()
try:
    P = torch.linalg.inv(G_dense)
except Exception as e:
    print(f"Inversion failed: {e}")
    # Fallback? No fallback for EASE closed form.
    raise e

print(f"Inversion done in {time.time() - start_time:.2f}s")

# Free G_dense
del G_dense
gc.collect()

# B = I - P * diag(1/diag(P))
# We don't need to compute full B. We can compute scores directly.
# Score = X @ B = X @ (I - P * diag(1/diag(P)))
#       = X - X @ P @ diag(1/diag(P))
#       = X - (X @ P) / diag(P)

# We need diag(P)
diag_P = torch.diag(P)
# We need P for inference.
# P is 74k x 74k. We can save it or use it for inference now.

#%% [code]
# 5. Inference on Test Set
print("Running Inference...")

# We only need to predict for test users.
# But X needs to be the full user history for those test users.
# Construct X_test_users: sparse matrix of shape (n_test_users, n_items)
# containing ALL history (train + val + test?)
# Wait, usually we predict based on Train history.
# So X_input should be the training interactions of the test users.

test_users = test_df['user'].unique()
test_user_idxs = [user2idx[u] for u in test_users if u in user2idx]

# Extract rows for test users from X (which contains train+val)
X_test_input = X[test_user_idxs] # Sparse CSR

# Convert to Tensor
# Doing X @ P is dense @ dense (or sparse @ dense).
# X_test_input (Sparse) @ P (Dense) -> Dense (n_test_users x n_items)
# n_test_users = 26k. 26k * 74k * 4 bytes = ~7.7 GB.
# This fits in memory (if P is cleared or managed).

# Let's process in batches to save memory
batch_size = 1000
n_batches = (len(test_user_idxs) + batch_size - 1) // batch_size

results = []

# Pre-compute 1/diag(P)
inv_diag_P = 1.0 / diag_P

print(f"Processing {len(test_user_idxs)} users in {n_batches} batches...")

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(test_user_idxs))
    
    batch_user_idxs = test_user_idxs[start:end]
    
    # Get input history for this batch
    # X is scipy sparse. Slicing is fast.
    x_batch_scipy = X[batch_user_idxs]
    x_batch = torch.from_numpy(x_batch_scipy.toarray()).to(device)
    
    # Compute Scores: S = X - (X @ P) * inv_diag_P
    # 1. X @ P
    xp = torch.matmul(x_batch, P)
    
    # 2. XP * inv_diag_P (broadcasting)
    xp_scaled = xp * inv_diag_P.unsqueeze(0)
    
    # 3. X - ...
    scores = x_batch - xp_scaled
    
    # Set score of already seen items to -inf (to exclude them)
    # x_batch is 1 where seen.
    scores[x_batch > 0] = -float('inf')
    
    # Top-K
    # We need to save top-K for ensemble. Let's save top 100 to allow good reranking.
    
    topk_scores, topk_indices = torch.topk(scores, k=100, dim=1)
    
    # Store results
    # We need to map back to original user/item IDs for the ensemble script?
    # Or keep indices. Indices are better for ensemble script.
    
    # We will save a dictionary: {user_idx: (top_item_idxs, top_scores)}
    for j, u_idx in enumerate(batch_user_idxs):
        results.append({
            'user_idx': u_idx,
            'items': topk_indices[j].cpu().numpy(),
            'scores': topk_scores[j].cpu().numpy()
        })
        
    if (i+1) % 10 == 0:
        print(f"Batch {i+1}/{n_batches} done")
        gc.collect()

# Save Results
print("Saving EASE results...")
with open(f'{OUTPUT_DIR}/ease_predictions.pkl', 'wb') as f:
    pickle.dump(results, f)

print("✅ EASE Training and Inference Complete.")
