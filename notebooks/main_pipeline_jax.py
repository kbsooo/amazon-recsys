#%% [markdown]
# # Amazon RecSys - SimGCL with JAX/FLAX
# 
# PyTorch 버전의 JAX/FLAX 구현입니다.
# **Note**: EASE는 행렬 연산이므로 JAX numpy만으로 구현 가능하며, 여기서는 SimGCL에 초점을 맞춥니다.

#%% [code]
import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Any, Callable
import time

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
edge_index_np = graph_data['edge_index'].numpy()  # [2, num_edges]
edge_weight_np = graph_data['cca_weight'].numpy()  # [num_edges]

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
        # Embeddings
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
        
        # Message passing: messages = all_emb[col] * edge_weight
        messages = all_emb[col] * edge_weight[:, None]
        
        # Aggregation: scatter_add
        # JAX doesn't have scatter_add like PyTorch, we use segment_sum
        # First, we need to sort by row indices
        total_nodes = self.n_users + self.n_items
        
        # Use jax.ops.segment_sum (requires sorted indices)
        # Alternative: use at[].add() with indices
        new_emb = jnp.zeros_like(all_emb)
        new_emb = new_emb.at[row].add(messages)
        
        return new_emb
    
    def __call__(self, edge_index, edge_weight, perturbed=False, training=False, rng=None):
        """Forward pass"""
        # Get initial embeddings
        user_emb_init = self.user_emb(jnp.arange(self.n_users))
        item_emb_init = self.item_emb(jnp.arange(self.n_items))
        all_emb = jnp.concatenate([user_emb_init, item_emb_init], axis=0)
        
        # Perturbation (for contrastive learning)
        if perturbed and training:
            if rng is None:
                raise ValueError("RNG required for perturbation")
            random_noise = random.normal(rng, all_emb.shape)
            # Normalize noise
            random_noise = random_noise / (jnp.linalg.norm(random_noise, axis=-1, keepdims=True) + 1e-10)
            # Add perturbation
            all_emb = all_emb + jnp.sign(all_emb) * random_noise * self.eps
        
        # Layer-wise propagation
        embs = [all_emb]
        for _ in range(self.n_layers):
            all_emb = self.graph_convolution(all_emb, edge_index, edge_weight)
            embs.append(all_emb)
        
        # Mean pooling across layers
        final_emb = jnp.mean(jnp.stack(embs), axis=0)
        
        # Split into user and item embeddings
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb

print("✅ LightGCN_SimGCL (FLAX) defined")

#%% [markdown]
# ## Part 3: Loss Functions

#%% [code]
def compute_bpr_loss(user_emb, item_emb, u_idx, pos_idx, neg_idx):
    """BPR Loss"""
    pos_scores = jnp.sum(user_emb[u_idx] * item_emb[pos_idx], axis=-1)
    neg_scores = jnp.sum(user_emb[u_idx] * item_emb[neg_idx], axis=-1)
    loss = -jnp.mean(jnp.log(jax.nn.sigmoid(pos_scores - neg_scores) + 1e-10))
    return loss

def compute_infonce_loss(emb_1, emb_2, temperature=0.2):
    """InfoNCE Contrastive Loss"""
    # Normalize
    emb_1 = emb_1 / (jnp.linalg.norm(emb_1, axis=-1, keepdims=True) + 1e-10)
    emb_2 = emb_2 / (jnp.linalg.norm(emb_2, axis=-1, keepdims=True) + 1e-10)
    
    # Positive pairs
    pos_score = jnp.sum(emb_1 * emb_2, axis=-1) / temperature
    
    # All pairs
    all_scores = jnp.matmul(emb_1, emb_2.T) / temperature
    
    # InfoNCE
    loss = -jnp.mean(jnp.log(jnp.exp(pos_score) / jnp.sum(jnp.exp(all_scores), axis=1)))
    
    return loss

print("✅ Loss functions defined")

#%% [markdown]
# ## Part 4: Training Setup

#%% [code]
print("\n" + "="*60)
print("🏋️ 학습 준비")
print("="*60)

# Hyperparameters
EMB_DIM = 64
N_LAYERS = 3
EPOCHS = 10  # Reduced for demo
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
dummy_edge_index = edge_index_jax
dummy_edge_weight = edge_weight_jax

variables = model.init(init_rng, dummy_edge_index, dummy_edge_weight, perturbed=False, training=False)
params = variables['params']

# Optimizer
tx = optax.adam(LR)
opt_state = tx.init(params)

# Create TrainState
class TrainState(train_state.TrainState):
    pass

state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx
)

print(f"✅ Model initialized")
print(f"  Parameters: {sum(x.size for x in jax.tree_leaves(params))}")

#%% [markdown]
# ## Part 5: Training Loop

#%% [code]
@jit
def train_step(state, u_idx, pos_idx, neg_idx, edge_index, edge_weight, rng):
    """Single training step"""
    
    def loss_fn(params):
        # Forward pass
        rng1, rng2, rng3 = random.split(rng, 3)
        
        user_emb, item_emb = model.apply(
            {'params': params},
            edge_index, edge_weight,
            perturbed=False, training=True
        )
        
        # BPR Loss
        bpr_loss = compute_bpr_loss(user_emb, item_emb, u_idx, pos_idx, neg_idx)
        
        # Contrastive Loss
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
        
        # Total loss
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

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    
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
    
    avg_loss = epoch_loss / n_batches
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("✅ Training complete")

#%% [markdown]
# ## Part 6: 추론

#%% [code]
print("\n" + "="*60)
print("🔮 추론")
print("="*60)

# Inference
user_emb, item_emb = model.apply(
    {'params': state.params},
    edge_index_jax, edge_weight_jax,
    perturbed=False, training=False
)

# Convert to numpy for easier manipulation
user_emb_np = np.array(user_emb)
item_emb_np = np.array(item_emb)

# Top-K recommendation
test_users = test_df['user'].unique()[:100]  # Sample for demo
recommendations = []

for u in test_users:
    if u not in user2idx:
        continue
    
    u_idx = user2idx[u]
    seen = user_train_items.get(u_idx, set())
    
    # Scores
    scores = user_emb_np[u_idx] @ item_emb_np.T
    scores[list(seen)] = -np.inf
    
    # Top-20
    top20_idx = np.argsort(scores)[-20:][::-1]
    recommendations.append((u, top20_idx.tolist()))

print(f"✅ Generated recommendations for {len(recommendations)} users")

#%% [markdown]
# ## 정리
# 
# JAX/FLAX를 사용한 SimGCL 구현을 완료했습니다.
# 
# **주요 차이점 (vs PyTorch)**:
# - **함수형 프로그래밍**: JAX는 순수 함수를 선호하며, 상태가 없습니다.
# - **명시적 RNG**: 난수 생성 시 RNG 키를 명시적으로 관리해야 합니다.
# - **JIT 컴파일**: `@jit` 데코레이터로 함수를 컴파일하여 성능을 향상시킵니다.
# - **Immutable 파라미터**: FLAX의 파라미터는 불변(immutable) 객체입니다.
# 
# **성능**: JAX는 XLA 컴파일을 통해 TPU/GPU에서 매우 빠른 속도를 제공합니다.
