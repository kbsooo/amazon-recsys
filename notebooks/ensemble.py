#%% [markdown]
# Ensemble: SimGCL + EASE
# Combines graph-based and item-based signals for robust recommendation.

#%% [code]
import numpy as np
import pandas as pd
import torch
import pickle
import os
import sys
import gc

# Add src to path to import model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# If running as notebook, __file__ might not exist.
if os.path.abspath('../src') not in sys.path:
    sys.path.append(os.path.abspath('../src'))

from model import LightGCN_SimGCL

# Configuration
DATA_DIR = '/kaggle/input/amazon' if os.path.exists('/kaggle/input/amazon') else 'data'
MODEL_PATH = 'models/simgcl_final.pt' # Or best.pt
EASE_PREDS_PATH = 'outputs/ease_predictions.pkl'
OUTPUT_PATH = 'outputs/ensemble_predictions.csv'

ALPHA = 0.5 # Weight for SimGCL (1-ALPHA for EASE)
TOP_K_CANDIDATES = 100 # Number of candidates to consider from each model

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

#%% [code]
# 1. Load Data & Mappings
print("Loading data and mappings...")
test_df = pd.read_csv(f'{DATA_DIR}/test_split.csv')

with open(f'{DATA_DIR}/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)
with open(f'{DATA_DIR}/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)
with open(f'{DATA_DIR}/user_k.pkl', 'rb') as f:
    user_k = pickle.load(f)
with open(f'{DATA_DIR}/user_train_items.pkl', 'rb') as f:
    user_train_items = pickle.load(f)

# Reverse mappings
idx2user = {v: k for k, v in user2idx.items()}
idx2item = {v: k for k, v in item2idx.items()}

#%% [code]
# 2. Load EASE Predictions
print(f"Loading EASE predictions from {EASE_PREDS_PATH}...")
with open(EASE_PREDS_PATH, 'rb') as f:
    ease_results = pickle.load(f)

# Convert to dictionary for fast lookup: user_idx -> {item_idx: score}
ease_scores_map = {}
for res in ease_results:
    u_idx = res['user_idx']
    items = res['items']
    scores = res['scores']
    
    # Normalize EASE scores per user (Min-Max)
    # EASE scores can be negative and large.
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s > 1e-6:
        norm_scores = (scores - min_s) / (max_s - min_s)
    else:
        norm_scores = np.zeros_like(scores) # All same
        
    ease_scores_map[u_idx] = {item: score for item, score in zip(items, norm_scores)}

print(f"Loaded EASE predictions for {len(ease_scores_map)} users.")

#%% [code]
# 3. Load SimGCL Model & Run Inference
print("Loading SimGCL model...")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
config = checkpoint['config']

model = LightGCN_SimGCL(
    config['n_users'],
    config['n_items'],
    config['emb_dim'],
    config['n_layers'],
    eps=config['eps']
).to(device)

model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Load Graph
graph_data = torch.load(f'{DATA_DIR}/train_graph.pt', map_location=device, weights_only=False)
edge_index = graph_data['edge_index'].to(device)
edge_weight = graph_data['cca_weight'].to(device)

print("Computing SimGCL embeddings...")
with torch.no_grad():
    user_emb, item_emb = model(edge_index, edge_weight, perturbed=False)

#%% [code]
# 4. Ensemble & Prediction
print("Running Ensemble Prediction...")

final_results = []
test_users = test_df['user'].unique()

# Prepare batch processing for SimGCL inference
test_user_idxs = [user2idx[u] for u in test_users if u in user2idx]
batch_size = 1024
n_batches = (len(test_user_idxs) + batch_size - 1) // batch_size

print(f"Processing {len(test_user_idxs)} users in {n_batches} batches...")

for i in range(n_batches):
    start = i * batch_size
    end = min((i + 1) * batch_size, len(test_user_idxs))
    batch_u_idxs = test_user_idxs[start:end]
    
    # SimGCL Scores
    batch_u_emb = user_emb[batch_u_idxs] # [B, Dim]
    # [B, Dim] @ [Dim, N_items] -> [B, N_items]
    batch_scores = torch.matmul(batch_u_emb, item_emb.t())
    
    # Mask seen items
    # This is slow to do per user in a batch loop if we iterate.
    # But we need to filter seen items.
    # Let's iterate users in this batch.
    
    for j, u_idx in enumerate(batch_u_idxs):
        # SimGCL Scores for this user
        s_scores = batch_scores[j] # [N_items]
        
        # Filter seen
        seen_items = user_train_items.get(u_idx, set())
        if seen_items:
            # Create mask? Or just set to -inf
            # Converting set to tensor indices is fast enough
            seen_tensor = torch.tensor(list(seen_items), device=device, dtype=torch.long)
            s_scores[seen_tensor] = -float('inf')
            
        # Top-K SimGCL Candidates
        # We take top 100 to match EASE
        s_topk_scores, s_topk_indices = torch.topk(s_scores, k=TOP_K_CANDIDATES)
        
        s_topk_indices_np = s_topk_indices.cpu().numpy()
        s_topk_scores_np = s_topk_scores.cpu().numpy()
        
        # Normalize SimGCL scores
        min_s = s_topk_scores_np.min()
        max_s = s_topk_scores_np.max()
        if max_s - min_s > 1e-6:
            s_norm_scores = (s_topk_scores_np - min_s) / (max_s - min_s)
        else:
            s_norm_scores = np.zeros_like(s_topk_scores_np)
            
        simgcl_map = {item: score for item, score in zip(s_topk_indices_np, s_norm_scores)}
        
        # Get EASE scores
        ease_map = ease_scores_map.get(u_idx, {})
        
        # Merge Candidates
        all_candidates = set(simgcl_map.keys()) | set(ease_map.keys())
        
        merged_scores = []
        for item in all_candidates:
            s_score = simgcl_map.get(item, 0.0) # If not in top-k, assume 0 (min)
            e_score = ease_map.get(item, 0.0)
            
            final_score = ALPHA * s_score + (1 - ALPHA) * e_score
            merged_scores.append((item, final_score))
            
        # Sort by final score
        merged_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select Top-N based on Rules
        K = user_k.get(u_idx, 0)
        
        # Rule: Max 50% of K
        num_recommend = int(K * 0.5)
        
        # K=1 Handling (Optional)
        # If K=1, num_recommend is 0.
        # If we wanted to force recommendation:
        # if num_recommend == 0 and K > 0: num_recommend = 1
        
        final_recs = []
        if num_recommend > 0:
            top_n = merged_scores[:num_recommend]
            final_recs = [item for item, score in top_n]
            
        # Add to results
        # We need to output for ALL test items for this user?
        # No, the submission format is: user, item, recommend(O/X)
        # We need to iterate over the test_df rows for this user.
        
        # Wait, the prediction loop structure in train_simgcl.py was:
        # Iterate test_df rows.
        # Here we are iterating users.
        # We should map back.
        
        # Let's store the recommended items set for this user
        rec_set = set(final_recs)
        
        # We can't easily map back to test_df rows here efficiently without grouping test_df.
        # Let's do that outside.
        # Store rec_set in a big dict: user_id -> set(item_ids)
        
        # But we need user_id string, not idx
        u_id_str = idx2user[u_idx]
        
        # Convert item indices to item strings
        rec_item_strs = {idx2item[i] for i in rec_set}
        
        # We will use a global dict to store recommendations
        # But we can't pass this to the outer loop easily if we process in batches.
        # Actually we can.
        
        # Let's just store (u_id_str, rec_item_strs) in a list
        final_results.append((u_id_str, rec_item_strs))

    if (i+1) % 10 == 0:
        print(f"Batch {i+1}/{n_batches} done")

# Convert results to dict
print("Generating submission file...")
user_rec_map = {u: recs for u, recs in final_results}

# Generate rows
submission_rows = []
# Group test_df by user to speed up?
# Or just iterate? Iterating 46k rows is fast.

for _, row in test_df.iterrows():
    u = row['user']
    i = row['item']
    
    recs = user_rec_map.get(u, set())
    recommend = 'O' if i in recs else 'X'
    
    submission_rows.append({'user': u, 'item': i, 'recommend': recommend})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Ensemble predictions saved to {OUTPUT_PATH}")

# Stats
rec_cnt = len(submission_df[submission_df['recommend'] == 'O'])
total_cnt = len(submission_df)
print(f"Total Recommendations: {rec_cnt}/{total_cnt} ({rec_cnt/total_cnt*100:.2f}%)")
