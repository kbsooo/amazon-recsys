#%% [markdown]
# # Amazon RecSys - SimGCL with JAX/FLAX (Improved)
# 
# PyTorch 버전의 JAX/FLAX 구현입니다 (개선 버전).
# - ✅ JAX API 업데이트 (v0.6.0+)
# - ✅ Loss & Metrics 시각화
# - ✅ Train/Val/Test 평가

#%% [code]
import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Any, Callable
import time
import matplotlib.pyplot as plt
from collections import defaultdict

# JAX imports
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from jax.tree_util import tree_map

# FLAX imports
import flax
from flax import linen as nn
from flax.training import train_state
import optax

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

SEED = 42
np.random.seed(SEED)

# Paths
DATA_DIR = 'data'
OUTPUT_DIR = 'outputs'
MODEL_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#%% [markdown]
# ## Part 1: 데이터 로드

#%% [code]
print("\n" + "="*60)
print("📊 데이터 로드")
print("="*60)

train_df = pd.read_csv(f'{DATA_DIR}/train_split.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_split.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test_split.csv')

with open(f'{DATA_DIR}/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)
with open(f'{DATA_DIR}/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)
with open(f'{DATA_DIR}/user_train_items.pkl', 'rb') as f:
    user_train_items = pickle.load(f)

n_users = len(user2idx)
n_items = len(item2idx)

print(f"Users: {n_users}, Items: {n_items}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# Load graph as numpy arrays
import torch
graph_data = torch.load(f'{DATA_DIR}/train_graph.pt', map_location='cpu', weights_only=False)
edge_index_np = graph_data['edge_index'].numpy()
edge_weight_np = graph_data['cca_weight'].numpy()

# Convert to JAX
edge_index_jax = jnp.array(edge_index_np)
edge_weight_jax = jnp.array(edge_weight_np)

print(f"Graph: {edge_index_jax.shape[1]} edges")

#%% [markdown]
# ## Part 2: LightGCN_SimGCL 모델 (FLAX)

#%% [code]
print("\n" + "="*60)
print("🧠 LightGCN_SimGCL (FLAX 구현)")
print("="*60)

class LightGCN_SimGCL(nn.Module):
    """SimGCL implementation using FLAX"""
    n_users: int
    n_items: int
    emb_dim: int = 64
    n_layers: int = 3
    eps: float = 0.1
    
    def setup(self):
        self.user_emb = nn.Embed(
            num_embeddings=self.n_users,
            features=self.emb_dim,
            embedding_init=nn.initializers.xavier_uniform()
        )
        self.item_emb = nn.Embed(
            num_embeddings=self.n_items,
            features=self.emb_dim,
            embedding_init=nn.initializers.xavier_uniform()
        )
    
    def graph_convolution(self, all_emb, edge_index, edge_weight):
        """Single layer of graph convolution"""
        row, col = edge_index[0], edge_index[1]
        messages = all_emb[col] * edge_weight[:, None]
        new_emb = jnp.zeros_like(all_emb)
        new_emb = new_emb.at[row].add(messages)
        return new_emb
    
    def __call__(self, edge_index, edge_weight, perturbed=False, training=False, rng=None):
        """Forward pass"""
        user_emb_init = self.user_emb(jnp.arange(self.n_users))
        item_emb_init = self.item_emb(jnp.arange(self.n_items))
        all_emb = jnp.concatenate([user_emb_init, item_emb_init], axis=0)
        
        if perturbed and training:
            if rng is None:
                raise ValueError("RNG required for perturbation")
            random_noise = random.normal(rng, all_emb.shape)
            random_noise = random_noise / (jnp.linalg.norm(random_noise, axis=-1, keepdims=True) + 1e-10)
            all_emb = all_emb + jnp.sign(all_emb) * random_noise * self.eps
        
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self.graph_convolution(all_emb, edge_index, edge_weight)
            embs.append(all_emb)
        
        final_emb = jnp.mean(jnp.stack(embs), axis=0)
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb

print("✅ LightGCN_SimGCL (FLAX) defined")

#%% [markdown]
# ## Part 3: Loss Functions & Evaluation

#%% [code]
def compute_bpr_loss(user_emb, item_emb, u_idx, pos_idx, neg_idx):
    """BPR Loss"""
    pos_scores = jnp.sum(user_emb[u_idx] * item_emb[pos_idx], axis=-1)
    neg_scores = jnp.sum(user_emb[u_idx] * item_emb[neg_idx], axis=-1)
    loss = -jnp.mean(jnp.log(jax.nn.sigmoid(pos_scores - neg_scores) + 1e-10))
    return loss

def compute_infonce_loss(emb_1, emb_2, temperature=0.2):
    """InfoNCE Contrastive Loss"""
    emb_1 = emb_1 / (jnp.linalg.norm(emb_1, axis=-1, keepdims=True) + 1e-10)
    emb_2 = emb_2 / (jnp.linalg.norm(emb_2, axis=-1, keepdims=True) + 1e-10)
    pos_score = jnp.sum(emb_1 * emb_2, axis=-1) / temperature
    all_scores = jnp.matmul(emb_1, emb_2.T) / temperature
    loss = -jnp.mean(jnp.log(jnp.exp(pos_score) / jnp.sum(jnp.exp(all_scores), axis=1)))
    return loss

def evaluate_metrics_jax(user_emb, item_emb, eval_df, user_train_items, k=20):
    """Recall, NDCG, Precision, MAP 계산"""
    user_emb_np = np.array(user_emb)
    item_emb_np = np.array(item_emb)
    
    user_metrics = defaultdict(lambda: {'hits': 0, 'total': 0, 'dcg': 0.0, 'idcg': 0.0, 'ap': 0.0})
    
    for u_idx in eval_df['user_idx'].unique():
        seen = user_train_items.get(u_idx, set())
        true_items = set(eval_df[eval_df['user_idx'] == u_idx]['item_idx'].values)
        
        scores = user_emb_np[u_idx] @ item_emb_np.T
        scores[list(seen)] = -np.inf
        
        topk_items = np.argsort(scores)[-k:][::-1].tolist()
        
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

print("✅ Loss & evaluation functions defined")

#%% [markdown]
# ## Part 4: Training Setup

#%% [code]
print("\n" + "="*60)
print("🏋️ 학습 준비")
print("="*60)

# Hyperparameters
EMB_DIM = 64
N_LAYERS = 3
EPOCHS = 50
BATCH_SIZE = 2048
LR = 0.001
LAMBDA_CL = 0.2
TEMPERATURE = 0.2
EPS = 0.1

# Initialize model
rng = random.PRNGKey(SEED)
rng, init_rng = random.split(rng)

model = LightGCN_SimGCL(
    n_users=n_users,
    n_items=n_items,
    emb_dim=EMB_DIM,
    n_layers=N_LAYERS,
    eps=EPS
)

# Initialize parameters
variables = model.init(init_rng, edge_index_jax, edge_weight_jax, perturbed=False, training=False)
params = variables['params']

# Optimizer
tx = optax.adam(LR)

# Create TrainState
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

# ✅ Fixed: Use jax.tree.leaves for JAX v0.6.0+
param_count = sum(x.size for x in jax.tree.leaves(params))
print(f"✅ Model initialized")
print(f"  Parameters: {param_count:,}")

#%% [markdown]
# ## Part 5: Training Loop with Visualization

#%% [code]
@jit
def train_step(state, u_idx, pos_idx, neg_idx, edge_index, edge_weight, rng):
    """Single training step"""
    
    def loss_fn(params):
        rng1, rng2 = random.split(rng)
        
        user_emb, item_emb = model.apply(
            {'params': params},
            edge_index, edge_weight,
            perturbed=False, training=True
        )
        
        bpr_loss = compute_bpr_loss(user_emb, item_emb, u_idx, pos_idx, neg_idx)
        
        u_emb_1, i_emb_1 = model.apply(
            {'params': params},
            edge_index, edge_weight,
            perturbed=True, training=True, rng=rng1
        )
        u_emb_2, i_emb_2 = model.apply(
            {'params': params},
            edge_index, edge_weight,
            perturbed=True, training=True, rng=rng2
        )
        
        cl_loss = compute_infonce_loss(u_emb_1[u_idx], u_emb_2[u_idx], TEMPERATURE) + \
                  compute_infonce_loss(i_emb_1[pos_idx], i_emb_2[pos_idx], TEMPERATURE)
        
        total_loss = bpr_loss + LAMBDA_CL * cl_loss
        
        return total_loss, (bpr_loss, cl_loss)
    
    (loss, (bpr_loss, cl_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    
    return state, loss, bpr_loss, cl_loss

print("\n" + "="*60)
print("🏋️ 학습 시작")
print("="*60)

# Prepare training data
pos_edges = jnp.array(train_df[['user_idx', 'item_idx']].values)
n_batches = (len(pos_edges) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Epochs: {EPOCHS}, Batches/Epoch: {n_batches}")

# History tracking
history = {
    'total_loss': [],
    'bpr_loss': [],
    'cl_loss': [],
    'val_recall': [],
    'val_ndcg': [],
    'val_precision': [],
    'val_map': []
}

best_val_recall = 0.0

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    epoch_bpr = 0.0
    epoch_cl = 0.0
    
    # Shuffle
    rng, shuffle_rng = random.split(rng)
    perm = random.permutation(shuffle_rng, len(pos_edges))
    pos_edges_shuffled = pos_edges[perm]
    
    for i in range(n_batches):
        batch_pos = pos_edges_shuffled[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        u_idx = batch_pos[:, 0]
        pos_idx = batch_pos[:, 1]
        
        # Negative sampling
        rng, neg_rng, step_rng = random.split(rng, 3)
        neg_idx = random.randint(neg_rng, (len(u_idx),), 0, n_items)
        
        # Train step
        state, loss, bpr_loss, cl_loss = train_step(
            state, u_idx, pos_idx, neg_idx,
            edge_index_jax, edge_weight_jax, step_rng
        )
        
        epoch_loss += loss
        epoch_bpr += bpr_loss
        epoch_cl += cl_loss
    
    avg_loss = epoch_loss / n_batches
    avg_bpr = epoch_bpr / n_batches
    avg_cl = epoch_cl / n_batches
    
    history['total_loss'].append(float(avg_loss))
    history['bpr_loss'].append(float(avg_bpr))
    history['cl_loss'].append(float(avg_cl))
    
    # Validation every 5 epochs
    if epoch % 5 == 0 or epoch == 1:
        user_emb, item_emb = model.apply(
            {'params': state.params},
            edge_index_jax, edge_weight_jax,
            perturbed=False, training=False
        )
        
        # Sample for speed (1000 users)
        val_sample = val_df[val_df['user_idx'].isin(val_df['user_idx'].unique()[:1000])]
        metrics = evaluate_metrics_jax(user_emb, item_emb, val_sample, user_train_items, k=20)
        
        history['val_recall'].append(metrics['Recall'])
        history['val_ndcg'].append(metrics['NDCG'])
        history['val_precision'].append(metrics['Precision'])
        history['val_map'].append(metrics['MAP'])
        
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} (BPR: {avg_bpr:.4f}, CL: {avg_cl:.4f}) | "
              f"Recall@20: {metrics['Recall']:.4f}, NDCG@20: {metrics['NDCG']:.4f}")
        
        if metrics['Recall'] > best_val_recall:
            best_val_recall = metrics['Recall']
            print(f"  💾 New best model (Recall@20: {best_val_recall:.4f})")
    else:
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} (BPR: {avg_bpr:.4f}, CL: {avg_cl:.4f})")

print(f"✅ Training complete (Best Val Recall@20: {best_val_recall:.4f})")

#%% [markdown]
# ## Part 6: 시각화

#%% [code]
print("\n" + "="*60)
print("📊 학습 곡선 시각화")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
ax1 = axes[0, 0]
ax1.plot(history['total_loss'], label='Total Loss', linewidth=2)
ax1.plot(history['bpr_loss'], label='BPR Loss', linewidth=2, alpha=0.7)
ax1.plot(history['cl_loss'], label='CL Loss', linewidth=2, alpha=0.7)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss Curves (JAX/FLAX)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Validation Recall & NDCG
ax2 = axes[0, 1]
eval_epochs = [1] + list(range(5, EPOCHS + 1, 5))
ax2.plot(eval_epochs, history['val_recall'], marker='o', label='Recall@20', linewidth=2)
ax2.plot(eval_epochs, history['val_ndcg'], marker='s', label='NDCG@20', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Score')
ax2.set_title('Validation: Recall & NDCG')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Validation Precision & MAP
ax3 = axes[1, 0]
ax3.plot(eval_epochs, history['val_precision'], marker='o', label='Precision@20', linewidth=2, color='green')
ax3.plot(eval_epochs, history['val_map'], marker='s', label='MAP@20', linewidth=2, color='orange')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Score')
ax3.set_title('Validation: Precision & MAP')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Loss components ratio
ax4 = axes[1, 1]
bpr_ratio = [b / (b + c) if (b + c) > 0 else 0 for b, c in zip(history['bpr_loss'], history['cl_loss'])]
ax4.plot(bpr_ratio, label='BPR Loss Ratio', linewidth=2, color='purple')
ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% Line')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('BPR / (BPR + CL)')
ax4.set_title('Loss Component Balance')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/jax_training_curves.png', dpi=150, bbox_inches='tight')
print(f"✅ Saved to {OUTPUT_DIR}/jax_training_curves.png")
plt.show()

#%% [markdown]
# ## Part 7: 최종 평가 (Train/Val/Test)

#%% [code]
print("\n" + "="*60)
print("📈 최종 평가 (All Splits)")
print("="*60)

# Get final embeddings
user_emb, item_emb = model.apply(
    {'params': state.params},
    edge_index_jax, edge_weight_jax,
    perturbed=False, training=False
)

# Evaluate on all splits
results = {}

for split_name, split_df in [('Train', train_df[:10000]), ('Val', val_df), ('Test', test_df[:5000])]:
    metrics = evaluate_metrics_jax(user_emb, item_emb, split_df, user_train_items, k=20)
    results[split_name] = metrics
    print(f"\n{split_name} Set @20:")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  NDCG:      {metrics['NDCG']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  MAP:       {metrics['MAP']:.4f}")

# Visualization: Bar chart comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics_list = ['Recall', 'NDCG', 'Precision', 'MAP']
x = np.arange(len(metrics_list))
width = 0.25

for i, split_name in enumerate(['Train', 'Val', 'Test']):
    values = [results[split_name][m] for m in metrics_list]
    ax.bar(x + i * width, values, width, label=split_name, alpha=0.8)

ax.set_xlabel('Metrics')
ax.set_ylabel('Score')
ax.set_title('Performance Comparison Across Splits (JAX/FLAX)')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics_list)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/jax_evaluation_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved to {OUTPUT_DIR}/jax_evaluation_comparison.png")
plt.show()

#%% [markdown]
# ## 정리
# 
# ✅ **개선 사항**:
# 1. JAX API 업데이트: `jax.tree_leaves` → `jax.tree.leaves` (v0.6.0+)
# 2. 다양한 loss & metrics 시각화: Total/BPR/CL Loss, Recall/NDCG/Precision/MAP
# 3. Train/Val/Test 평가 및 비교 차트
# 
# **JAX/FLAX 장점**:
# - TPU에서 즉시 실행 가능
# - JIT 컴파일로 빠른 속도
# - 함수형 프로그래밍으로 디버깅 용이
