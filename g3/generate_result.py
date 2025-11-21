import pandas as pd
import torch
import numpy as np
import pickle
import os
import argparse
import sys

# Define Model Class (Must match training)
import torch.nn as nn

class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)
        
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

def load_artifacts(base_dir):
    with open(os.path.join(base_dir, 'user_mapper.pkl'), 'rb') as f:
        user_mapper = pickle.load(f)
    with open(os.path.join(base_dir, 'item_mapper.pkl'), 'rb') as f:
        item_mapper = pickle.load(f)
    
    # Load Train Data for History
    train_df = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))
    user_history = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    return user_mapper, item_mapper, user_history

def main():
    parser = argparse.ArgumentParser(description='Generate Recommendations')
    parser.add_argument('input_file', type=str, help='Input CSV file with user, item columns')
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load Artifacts
    user_mapper, item_mapper, user_history = load_artifacts(base_dir)
    n_users = len(user_mapper)
    n_items = len(item_mapper)
    
    # MPS는 Sparse Tensor 연산(aten::addmm) 지원이 미비하여 CPU 사용
    device = torch.device('cpu')
    print(f"Using device: {device}")
    model = LightGCN(n_users, n_items, emb_dim=64, n_layers=2) # Match training config
    model.load_state_dict(torch.load(os.path.join(base_dir, 'lightgcn_model.pt'), map_location=device))
    model.to(device)
    model.eval()
    
    # Precompute Embeddings (Since we don't need graph for inference if we use learned embeddings)
    # Wait, LightGCN inference needs the graph to propagate embeddings?
    # Yes, strictly speaking, the final embedding is the result of propagation.
    # So we need to reconstruct the adjacency matrix or save the final embeddings.
    # Reconstructing adj matrix is expensive.
    # Better to save final embeddings during training?
    # Or just re-run forward pass once.
    
    # Reconstruct Adj Matrix (Simplified for inference)
    # We need the SAME adj matrix as training? Or full graph?
    # Usually full graph (train + test) if transductive, but here we only have train graph.
    # We will use the train graph.
    
    print("Reconstructing Graph for Inference...")
    train_df = pd.read_csv(os.path.join(base_dir, 'train_data.csv'))
    
    # Copy-paste create_adj_matrix logic or import it
    import scipy.sparse as sp
    u = train_df['user_idx'].values
    i = train_df['item_idx'].values
    user_np = np.array(u)
    item_np = np.array(i)
    ratings = np.ones_like(user_np, dtype=np.float32)
    n_nodes = n_users + n_items
    tmp_adj = sp.coo_matrix((ratings, (user_np, item_np + n_users)), shape=(n_nodes, n_nodes))
    adj_mat = tmp_adj + tmp_adj.T
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj_mat).dot(d_mat)
    norm_adj = norm_adj.tocoo()
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    shape = torch.Size(norm_adj.shape)
    adj_matrix = torch.sparse_coo_tensor(indices, values, shape).to(device)
    
    print("Computing Embeddings...")
    with torch.no_grad():
        users_emb, items_emb = model(adj_matrix)
    
    # Process Input File
    input_df = pd.read_csv(args.input_file)
    # Check columns
    if 'user' not in input_df.columns or 'item' not in input_df.columns:
        print("Error: Input file must have 'user' and 'item' columns")
        return

    results = []
    recommends_count = 0
    not_recommends_count = 0
    
    print("Processing queries...")
    # Group by user to optimize top-k retrieval
    # But input file might be random order.
    # We can cache user recommendations.
    
    user_recs_cache = {}
    
    for idx, row in input_df.iterrows():
        user_id = row['user']
        item_id = row['item']
        
        # Map IDs
        u_idx = user_mapper.get(user_id)
        i_idx = item_mapper.get(item_id)
        
        rec_status = 'X'
        
        if u_idx is not None:
            if u_idx not in user_recs_cache:
                # Determine K
                history_count = len(user_history.get(u_idx, []))
                if history_count <= 10:
                    k = 2
                else:
                    k = int(history_count * 0.5)
                    if k < 1: k = 1
                
                # Get Top-K
                u_e = users_emb[u_idx]
                scores = torch.matmul(items_emb, u_e)
                
                # Mask seen items?
                # Usually we don't recommend items already bought.
                seen = user_history.get(u_idx, [])
                scores[seen] = -float('inf')
                
                _, top_k = torch.topk(scores, k)
                user_recs_cache[u_idx] = set(top_k.cpu().numpy())
            
            # Check if item is in recommendations
            if i_idx is not None and i_idx in user_recs_cache[u_idx]:
                rec_status = 'O'
        
        results.append(rec_status)
        if rec_status == 'O':
            recommends_count += 1
        else:
            not_recommends_count += 1
            
    # Output
    print("====================")
    print("user   item   recommend")
    for i, row in input_df.iterrows():
        print(f"{row['user']}   {row['item']}    {results[i]}")
        if i >= 10: # Limit output for large files
            print(" ...")
            break
    print("====================")
    print(f"Total recommends = {recommends_count}/{len(input_df)}")
    print(f"Total not recommends = {not_recommends_count}/{len(input_df)}")

if __name__ == "__main__":
    main()
