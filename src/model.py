#%%
"""
모델 정의 - Amazon RecSys GNN
LightGCN 베이스 모델 + Rating Prediction 모델 + 앙상블
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
print("="*60)
print("LightGCN 모델 정의")
print("="*60)

class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    구조 학습(Structure Learning)에 집중하는 모델.
    BPR Loss로 학습하여 이진 추천 태스크에 활용.
    """
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        # Embedding Layers (Xavier initialization)
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
    
    def forward(self, edge_index, edge_weight):
        """
        Graph Convolution 수행
        
        Args:
            edge_index: [2, num_edges] - Edge Index
            edge_weight: [num_edges] - Normalized Edge Weights
        
        Returns:
            user_emb: [n_users, emb_dim]
            item_emb: [n_items, emb_dim]
        """
        # 초기 임베딩
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        
        # Layer-wise Propagation
        for _ in range(self.n_layers):
            row, col = edge_index
            # Weighted Message Passing
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            
            # Aggregation (scatter_add)
            all_emb = torch.zeros_like(all_emb).scatter_add(
                0, row.unsqueeze(1).expand(-1, self.emb_dim), messages
            )
            embs.append(all_emb)
        
        # Layer Combination (Mean pooling)
        final_emb = torch.mean(torch.stack(embs), dim=0)
        
        # User/Item 분리
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def predict(self, user_idx, item_idx, edge_index, edge_weight):
        """
        유저-아이템 쌍에 대한 예측 점수
        
        Args:
            user_idx: [batch_size] or single user
            item_idx: [batch_size] or single item
        
        Returns:
            scores: [batch_size]
        """
        user_emb, item_emb = self(edge_index, edge_weight)
        scores = (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)
        return scores

print("✅ LightGCN 클래스 정의 완료")

#%%
print("\n" + "="*60)
print("LightGCN + Rating Prediction 모델 정의")
print("="*60)

class LightGCN_Rating(nn.Module):
    """
    LightGCN + Rating Prediction Head
    
    구조 학습 + 평점 예측을 동시에 수행.
    Multi-task Learning으로 일반화 성능 향상.
    """
    def __init__(self, n_users, n_items, emb_dim=64, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        # Embedding Layers
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        
        # Rating Prediction MLP
        self.rating_mlp = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def forward(self, edge_index, edge_weight):
        """Graph Convolution"""
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        embs = [all_emb]
        
        for _ in range(self.n_layers):
            row, col = edge_index
            messages = all_emb[col] * edge_weight.unsqueeze(1)
            all_emb = torch.zeros_like(all_emb).scatter_add(
                0, row.unsqueeze(1).expand(-1, self.emb_dim), messages
            )
            embs.append(all_emb)
        
        final_emb = torch.mean(torch.stack(embs), dim=0)
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def predict_rating(self, user_idx, item_idx, edge_index, edge_weight):
        """
        평점 예측 (1.0 ~ 5.0 범위)
        
        Returns:
            ratings: [batch_size] - 예측 평점
        """
        user_emb, item_emb = self(edge_index, edge_weight)
        interaction = user_emb[user_idx] * item_emb[item_idx]
        rating_logit = self.rating_mlp(interaction).squeeze(-1)
        
        # Sigmoid로 [0, 1] 변환 후 [0.5, 5.0] 범위로 스케일링
        rating = torch.sigmoid(rating_logit) * 4.5 + 0.5
        return rating
    
    def predict(self, user_idx, item_idx, edge_index, edge_weight):
        """구조 기반 점수 (BPR용)"""
        user_emb, item_emb = self(edge_index, edge_weight)
        scores = (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)
        return scores

print("✅ LightGCN_Rating 클래스 정의 완료")

#%%
print("\n" + "="*60)
print("LightGCN + SimGCL (Contrastive Learning) 모델 정의")
print("="*60)

class LightGCN_SimGCL(nn.Module):
    """
    LightGCN with Simple Graph Contrastive Learning (SimGCL)
    
    희소한 데이터에서 강건한 임베딩을 학습하기 위해 Contrastive Learning 적용.
    학습 시 임베딩에 노이즈를 추가하여 데이터 증강 효과를 얻습니다.
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
        Graph Convolution 수행
        
        Args:
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
            perturbed: True이면 노이즈가 섞인 임베딩 반환
        
        Returns:
            user_emb: [n_users, emb_dim]
            item_emb: [n_items, emb_dim]
        """
        # 초기 임베딩
        all_emb = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)
        
        # Perturbation (노이즈 추가)
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
        두 개의 perturbed view 생성 (Contrastive Learning용)
        """
        u_emb_1, i_emb_1 = self.forward(edge_index, edge_weight, perturbed=True)
        u_emb_2, i_emb_2 = self.forward(edge_index, edge_weight, perturbed=True)
        return u_emb_1, i_emb_1, u_emb_2, i_emb_2
    
    def predict(self, user_idx, item_idx, edge_index, edge_weight):
        """예측 점수 (추론 시 사용, perturbed=False)"""
        user_emb, item_emb = self.forward(edge_index, edge_weight, perturbed=False)
        scores = (user_emb[user_idx] * item_emb[item_idx]).sum(dim=-1)
        return scores

print("✅ LightGCN_SimGCL 클래스 정의 완료")

#%%
print("\n" + "="*60)
print("InfoNCE Loss (Contrastive Loss) 함수 정의")
print("="*60)

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

print("✅ InfoNCE Loss 함수 정의 완료")

#%%
print("\n" + "="*60)
print("모델 테스트")
print("="*60)

# 더미 데이터로 테스트
if __name__ == "__main__":
    # 파라미터
    n_users, n_items = 1000, 500
    emb_dim = 32
    n_layers = 2
    
    # 더미 그래프
    num_edges = 5000
    edge_index = torch.randint(0, n_users + n_items, (2, num_edges))
    edge_weight = torch.rand(num_edges)
    
    # LightGCN 테스트
    print("\n[LightGCN 테스트]")
    model_cca = LightGCN(n_users, n_items, emb_dim, n_layers)
    user_emb, item_emb = model_cca(edge_index, edge_weight)
    print(f"  User Embeddings: {user_emb.shape}")
    print(f"  Item Embeddings: {item_emb.shape}")
    
    # SimGCL 테스트
    print("\n[LightGCN_SimGCL 테스트]")
    model_simgcl = LightGCN_SimGCL(n_users, n_items, emb_dim, n_layers)
    model_simgcl.train()
    
    # Normal forward
    user_emb, item_emb = model_simgcl(edge_index, edge_weight, perturbed=False)
    print(f"  User Embeddings: {user_emb.shape}")
    print(f"  Item Embeddings: {item_emb.shape}")
    
    # Perturbed views
    u1, i1, u2, i2 = model_simgcl.get_perturbed_embeddings(edge_index, edge_weight)
    print(f"  Perturbed View 1 - Users: {u1.shape}, Items: {i1.shape}")
    print(f"  Perturbed View 2 - Users: {u2.shape}, Items: {i2.shape}")
    
    # InfoNCE Loss test
    cl_loss_u = compute_infonce_loss(u1[:100], u2[:100])
    cl_loss_i = compute_infonce_loss(i1[:100], i2[:100])
    print(f"  User Contrastive Loss: {cl_loss_u.item():.4f}")
    print(f"  Item Contrastive Loss: {cl_loss_i.item():.4f}")
    
    print("\n✅ 모델 정의 및 테스트 완료!")
