#%% [markdown]
# # SimGCL JAX - 하이퍼파라미터 튜닝 (Optuna)
# 
# Optuna를 사용하여 SimGCL 모델의 하이퍼파라미터를 자동으로 최적화합니다.

#%% [code]
import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Any, Callable
import time
from collections import defaultdict

# JAX imports
import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap

# FLAX imports
import flax
from flax import linen as nn
from flax.training import train_state
import optax

# Optuna
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

print(f"JAX version: {jax.__version__}")
print(f"Optuna version: {optuna.__version__}")

SEED = 42
np.random.seed(SEED)

# ✅ Kaggle 환경 감지 및 경로 설정
IS_KAGGLE = os.path.exists('/kaggle/input')

if IS_KAGGLE:
    print("🎯 Running on Kaggle")
    DATA_DIR = '/kaggle/input/amazon'
    WORKING_DIR = '/kaggle/working'
    OUTPUT_DIR = os.path.join(WORKING_DIR, 'outputs')
    MODEL_DIR = os.path.join(WORKING_DIR, 'models')
else:
    print("💻 Running locally")
    DATA_DIR = 'data'
    WORKING_DIR = '.'
    OUTPUT_DIR = 'outputs'
    MODEL_DIR = 'models'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

#%% [markdown]
# ## 데이터 로드

#%% [code]
print("\n" + "="*60)
print("📊 데이터 로드")
print("="*60)

train_df = pd.read_csv(f'{DATA_DIR}/train_split.csv')
val_df = pd.read_csv(f'{DATA_DIR}/val_split.csv')

with open(f'{DATA_DIR}/user2idx.pkl', 'rb') as f:
    user2idx = pickle.load(f)
with open(f'{DATA_DIR}/item2idx.pkl', 'rb') as f:
    item2idx = pickle.load(f)
with open(f'{DATA_DIR}/user_train_items.pkl', 'rb') as f:
    user_train_items = pickle.load(f)

n_users = len(user2idx)
n_items = len(item2idx)

print(f"Users: {n_users}, Items: {n_items}")
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Load graph
import torch
graph_data = torch.load(f'{DATA_DIR}/train_graph.pt', map_location='cpu', weights_only=False)
edge_index_np = graph_data['edge_index'].numpy()
edge_weight_np = graph_data['cca_weight'].numpy()

edge_index_jax = jnp.array(edge_index_np)
edge_weight_jax = jnp.array(edge_weight_np)

#%% [markdown]
# ## 모델 및 함수 정의

#%% [code]
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
        row, col = edge_index[0], edge_index[1]
        messages = all_emb[col] * edge_weight[:, None]
        new_emb = jnp.zeros_like(all_emb)
        new_emb = new_emb.at[row].add(messages)
        return new_emb
    
    def __call__(self, edge_index, edge_weight, perturbed=False, training=False, rng=None):
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
        return final_emb[:self.n_users], final_emb[self.n_users:]

def compute_bpr_loss(user_emb, item_emb, u_idx, pos_idx, neg_idx):
    pos_scores = jnp.sum(user_emb[u_idx] * item_emb[pos_idx], axis=-1)
    neg_scores = jnp.sum(user_emb[u_idx] * item_emb[neg_idx], axis=-1)
    loss = -jnp.mean(jnp.log(jax.nn.sigmoid(pos_scores - neg_scores) + 1e-10))
    return loss

def compute_infonce_loss(emb_1, emb_2, temperature=0.2):
    emb_1 = emb_1 / (jnp.linalg.norm(emb_1, axis=-1, keepdims=True) + 1e-10)
    emb_2 = emb_2 / (jnp.linalg.norm(emb_2, axis=-1, keepdims=True) + 1e-10)
    pos_score = jnp.sum(emb_1 * emb_2, axis=-1) / temperature
    all_scores = jnp.matmul(emb_1, emb_2.T) / temperature
    loss = -jnp.mean(jnp.log(jnp.exp(pos_score) / jnp.sum(jnp.exp(all_scores), axis=1)))
    return loss

def evaluate_recall(user_emb, item_emb, eval_df, user_train_items, k=20):
    """빠른 Recall@K 계산"""
    user_emb_np = np.array(user_emb)
    item_emb_np = np.array(item_emb)
    
    hits = 0
    total = 0
    
    for u_idx in eval_df['user_idx'].unique()[:1000]:  # 샘플링
        seen = user_train_items.get(u_idx, set())
        true_items = set(eval_df[eval_df['user_idx'] == u_idx]['item_idx'].values)
        
        scores = user_emb_np[u_idx] @ item_emb_np.T
        scores[list(seen)] = -np.inf
        
        topk_items = np.argsort(scores)[-k:][::-1].tolist()
        
        hits += len(set(topk_items) & true_items)
        total += len(true_items)
    
    return hits / total if total > 0 else 0.0

#%% [markdown]
# ## Optuna Objective 함수

#%% [code]
def objective(trial):
    """Optuna objective function"""
    
    # ✅ 하이퍼파라미터 샘플링
    emb_dim = trial.suggest_categorical('emb_dim', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 2, 4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    lambda_cl = trial.suggest_float('lambda_cl', 0.05, 0.5)
    temperature = trial.suggest_float('temperature', 0.1, 0.5)
    eps = trial.suggest_float('eps', 0.05, 0.2)
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}: EMB={emb_dim}, LAYERS={n_layers}, LR={lr:.4f}, "
          f"LAMBDA_CL={lambda_cl:.3f}, TEMP={temperature:.2f}, EPS={eps:.2f}")
    print('='*60)
    
    # 모델 초기화
    rng = random.PRNGKey(SEED + trial.number)
    rng, init_rng = random.split(rng)
    
    model = LightGCN_SimGCL(
        n_users=n_users,
        n_items=n_items,
        emb_dim=emb_dim,
        n_layers=n_layers,
        eps=eps
    )
    
    variables = model.init(init_rng, edge_index_jax, edge_weight_jax, perturbed=False, training=False)
    params = variables['params']
    
    tx = optax.adam(lr)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # JIT 컴파일된 train_step
    @jit
    def train_step(state, u_idx, pos_idx, neg_idx, rng):
        def loss_fn(params):
            rng1, rng2 = random.split(rng)
            
            user_emb, item_emb = model.apply(
                {'params': params},
                edge_index_jax, edge_weight_jax,
                perturbed=False, training=True
            )
            
            bpr_loss = compute_bpr_loss(user_emb, item_emb, u_idx, pos_idx, neg_idx)
            
            u_emb_1, _ = model.apply(
                {'params': params},
                edge_index_jax, edge_weight_jax,
                perturbed=True, training=True, rng=rng1
            )
            u_emb_2, _ = model.apply(
                {'params': params},
                edge_index_jax, edge_weight_jax,
                perturbed=True, training=True, rng=rng2
            )
            
            cl_loss = compute_infonce_loss(u_emb_1[u_idx], u_emb_2[u_idx], temperature)
            total_loss = bpr_loss + lambda_cl * cl_loss
            
            return total_loss
        
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        
        return state, loss
    
    # 학습 데이터 준비
    pos_edges = jnp.array(train_df[['user_idx', 'item_idx']].values)
    BATCH_SIZE = 2048
    n_batches = (len(pos_edges) + BATCH_SIZE - 1) // BATCH_SIZE
    MAX_EPOCHS = 30
    
    best_recall = 0.0
    
    for epoch in range(1, MAX_EPOCHS + 1):
        # Shuffle
        rng, shuffle_rng = random.split(rng)
        perm = random.permutation(shuffle_rng, len(pos_edges))
        pos_edges_shuffled = pos_edges[perm]
        
        # Training
        for i in range(n_batches):
            batch_pos = pos_edges_shuffled[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            u_idx = batch_pos[:, 0]
            pos_idx = batch_pos[:, 1]
            
            rng, neg_rng, step_rng = random.split(rng, 3)
            neg_idx = random.randint(neg_rng, (len(u_idx),), 0, n_items)
            
            state, loss = train_step(state, u_idx, pos_idx, neg_idx, step_rng)
        
        # Validation (5 epoch마다)
        if epoch % 5 == 0:
            user_emb, item_emb = model.apply(
                {'params': state.params},
                edge_index_jax, edge_weight_jax,
                perturbed=False, training=False
            )
            
            recall = evaluate_recall(user_emb, item_emb, val_df, user_train_items, k=20)
            
            print(f"  Epoch {epoch}/{MAX_EPOCHS} | Recall@20: {recall:.4f}")
            
            if recall > best_recall:
                best_recall = recall
            
            # ✅ Pruning (조기 종료)
            trial.report(recall, epoch)
            if trial.should_prune():
                print(f"  ✂️ Trial pruned at epoch {epoch}")
                raise optuna.TrialPruned()
    
    return best_recall

#%% [markdown]
# ## Optuna Study 실행

#%% [code]
print("\n" + "="*60)
print("🔍 하이퍼파라미터 튜닝 시작")
print("="*60)

# Study 생성
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED),
    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
)

# 최적화 실행
N_TRIALS = 50
study.optimize(objective, n_trials=N_TRIALS, timeout=None)

#%% [markdown]
# ## 결과 분석

#%% [code]
print("\n" + "="*60)
print("📊 튜닝 결과")
print("="*60)

print("\n🏆 Best Trial:")
print(f"  Recall@20: {study.best_value:.4f}")
print(f"\n  Best Hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")

# 결과 저장
results_df = study.trials_dataframe()
results_df.to_csv(f'{OUTPUT_DIR}/optuna_results.csv', index=False)
print(f"\n💾 Results saved to {OUTPUT_DIR}/optuna_results.csv")

# Visualization
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Optimization history
    ax1 = axes[0]
    ax1.plot([t.number for t in study.trials], [t.value for t in study.trials if t.value is not None], 'o-')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Recall@20')
    ax1.set_title('Optimization History')
    ax1.grid(True, alpha=0.3)
    
    # Param importances
    ax2 = axes[1]
    importances = optuna.importance.get_param_importances(study)
    ax2.barh(list(importances.keys()), list(importances.values()))
    ax2.set_xlabel('Importance')
    ax2.set_title('Hyperparameter Importances')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/optuna_analysis.png', dpi=150, bbox_inches='tight')
    print(f"💾 Visualization saved to {OUTPUT_DIR}/optuna_analysis.png")
except Exception as e:
    print(f"⚠️ Visualization failed: {e}")

print("\n✅ 튜닝 완료!")
