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
    
    # 예측 테스트
    test_users = torch.tensor([0, 1, 2])
    test_items = torch.tensor([10, 20, 30])
    scores = model_cca.predict(test_users, test_items, edge_index, edge_weight)
    print(f"  Prediction Scores: {scores}")
    
    # LightGCN_Rating 테스트
    print("\n[LightGCN_Rating 테스트]")
    model_ccb = LightGCN_Rating(n_users, n_items, emb_dim, n_layers)
    user_emb, item_emb = model_ccb(edge_index, edge_weight)
    print(f"  User Embeddings: {user_emb.shape}")
    print(f"  Item Embeddings: {item_emb.shape}")
    
    # 평점 예측 테스트
    ratings = model_ccb.predict_rating(test_users, test_items, edge_index, edge_weight)
    print(f"  Predicted Ratings: {ratings}")
    print(f"    범위: [{ratings.min():.2f}, {ratings.max():.2f}]")
    
    print("\n✅ 모델 정의 및 테스트 완료!")
