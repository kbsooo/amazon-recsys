"""
GNN 모델 구현

비교 실험을 위한 여러 GNN 모델들:
1. LightGCN - 주력 모델
2. NGCF - 비교 baseline
3. Simple GCN - 기본 baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class LightGCN(nn.Module):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
    
    특징:
    - Non-linear activation 제거
    - Feature transformation 제거
    - 순수 neighborhood aggregation만 수행
    - Layer-wise embedding을 평균하여 최종 embedding 생성
    """
    
    def __init__(self, n_users: int, n_items: int, 
                 embedding_dim: int = 64, 
                 n_layers: int = 3,
                 device: str = 'cpu'):
        """
        Args:
            n_users: 사용자 수
            n_items: 아이템 수
            embedding_dim: Embedding 차원
            n_layers: GCN layer 수
            device: 디바이스 ('cpu', 'cuda', 'mps')
        """
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        
        # User and Item Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Xavier initialization
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def compute_graph_convolution(self, edge_index: torch.Tensor, 
                                  edge_weight: torch.Tensor = None):
        """
        Graph convolution을 통한 embedding 전파
        
        Args:
            edge_index: Edge indices [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)
        
        Returns:
            (user_embeddings, item_embeddings): 각 layer의 embedding들을 평균한 최종 embedding
        """
        # 초기 embedding
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        
        # Concatenate user and item embeddings
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        # Store embeddings from each layer
        embs = [all_emb]
        
        # Graph convolution layers
        for layer in range(self.n_layers):
            all_emb = self._propagate(edge_index, all_emb, edge_weight)
            embs.append(all_emb)
        
        # Average all layer embeddings
        final_emb = torch.mean(torch.stack(embs, dim=0), dim=0)
        
        # Split back into user and item embeddings
        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]
        
        return user_final, item_final
    
    def _propagate(self, edge_index: torch.Tensor, 
                   embeddings: torch.Tensor,
                   edge_weight: torch.Tensor = None) -> torch.Tensor:
        """
        단일 layer propagation (simplified)
        
        Args:
            edge_index: Edge indices
            embeddings: Current embeddings
            edge_weight: Edge weights (optional)
        
        Returns:
            Updated embeddings
        """
        # Compute degree normalization
        row, col = edge_index
        deg = degree(row, embeddings.size(0), dtype=embeddings.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalize
        if edge_weight is None:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # Message passing: aggregate neighbors
        out = torch.zeros_like(embeddings)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += norm[i] * embeddings[src]
        
        return out
    
    def forward(self, users: torch.Tensor, items: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        Forward pass
        
        Args:
            users: User indices [batch_size]
            items: Item indices [batch_size] 
            edge_index: Graph edge indices
            edge_weight: Edge weights (optional)
        
        Returns:
            Predicted scores [batch_size]
        """
        # Get final embeddings
        user_emb, item_emb = self.compute_graph_convolution(edge_index, edge_weight)
        
        # Get embeddings for specific users and items
        user_emb_batch = user_emb[users]
        item_emb_batch = item_emb[items]
        
        # Dot product
        scores = (user_emb_batch * item_emb_batch).sum(dim=1)
        
        return scores
    
    def get_all_embeddings(self, edge_index: torch.Tensor, 
                          edge_weight: torch.Tensor = None):
        """모든 user와 item의 최종 embedding 반환"""
        return self.compute_graph_convolution(edge_index, edge_weight)


class NGCF(nn.Module):
    """
    Neural Graph Collaborative Filtering
    
    LightGCN보다 복잡한 구조:
    - Feature transformation 포함
    - Non-linear activation 포함
    - Message dropout 포함
    """
    
    def __init__(self, n_users: int, n_items: int,
                 embedding_dim: int = 64,
                 n_layers: int = 3,
                 hidden_dims: list = None,
                 dropout: float = 0.1,
                 device: str = 'cpu'):
        """
        Args:
            n_users: 사용자 수
            n_items: 아이템 수
            embedding_dim: Embedding 차원
            n_layers: GCN layer 수
            hidden_dims: 각 layer의 hidden dimension (None이면 모두 embedding_dim)
            dropout: Dropout rate
            device: 디바이스
        """
        super(NGCF, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        self.dropout = dropout
        
        if hidden_dims is None:
            hidden_dims = [embedding_dim] * n_layers
        self.hidden_dims = hidden_dims
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # GCN layers
        self.W1_list = nn.ModuleList()
        self.W2_list = nn.ModuleList()
        
        input_dim = embedding_dim
        for hidden_dim in hidden_dims:
            self.W1_list.append(nn.Linear(input_dim, hidden_dim))
            self.W2_list.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
    def compute_graph_convolution(self, edge_index: torch.Tensor,
                                  edge_weight: torch.Tensor = None):
        """Graph convolution with feature transformation"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        embs = [all_emb]
        
        for layer in range(self.n_layers):
            # Message passing with transformation
            all_emb = self._ngcf_layer(
                edge_index, all_emb, 
                self.W1_list[layer], self.W2_list[layer],
                edge_weight
            )
            # Activation
            all_emb = F.leaky_relu(all_emb)
            # Dropout
            all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            
            embs.append(all_emb)
        
        # Concatenate all layers
        final_emb = torch.cat(embs, dim=1)
        
        user_final = final_emb[:self.n_users]
        item_final = final_emb[self.n_users:]
        
        return user_final, item_final
    
    def _ngcf_layer(self, edge_index: torch.Tensor, embeddings: torch.Tensor,
                   W1: nn.Linear, W2: nn.Linear, edge_weight: torch.Tensor = None):
        """NGCF layer with bi-interaction"""
        row, col = edge_index
        deg = degree(row, embeddings.size(0), dtype=embeddings.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if edge_weight is None:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # Aggregate
        aggregated = torch.zeros_like(embeddings)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            aggregated[dst] += norm[i] * embeddings[src]
        
        # Transform
        out = W1(embeddings + aggregated) + W2(embeddings * aggregated)
        
        return out
    
    def forward(self, users: torch.Tensor, items: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """Forward pass"""
        user_emb, item_emb = self.compute_graph_convolution(edge_index, edge_weight)
        
        user_emb_batch = user_emb[users]
        item_emb_batch = item_emb[items]
        
        scores = (user_emb_batch * item_emb_batch).sum(dim=1)
        
        return scores
    
    def get_all_embeddings(self, edge_index: torch.Tensor,
                          edge_weight: torch.Tensor = None):
        """모든 embedding 반환"""
        return self.compute_graph_convolution(edge_index, edge_weight)


class SimpleGCN(nn.Module):
    """
    Simple GCN Baseline
    
    가장 기본적인 GCN 구조
    """
    
    def __init__(self, n_users: int, n_items: int,
                 embedding_dim: int = 64,
                 n_layers: int = 2,
                 device: str = 'cpu'):
        """
        Args:
            n_users: 사용자 수
            n_items: 아이템 수
            embedding_dim: Embedding 차원
            n_layers: GCN layer 수
            device: 디바이스
        """
        super(SimpleGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.device = device
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Linear layers
        self.layers = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(n_layers)
        ])
        
    def compute_graph_convolution(self, edge_index: torch.Tensor,
                                  edge_weight: torch.Tensor = None):
        """Simple GCN convolution"""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        for layer_idx, layer in enumerate(self.layers):
            # Propagate
            all_emb = self._propagate(edge_index, all_emb, edge_weight)
            # Transform
            all_emb = layer(all_emb)
            # Activation (except last layer)
            if layer_idx < len(self.layers) - 1:
                all_emb = F.relu(all_emb)
        
        user_final = all_emb[:self.n_users]
        item_final = all_emb[self.n_users:]
        
        return user_final, item_final
    
    def _propagate(self, edge_index: torch.Tensor, embeddings: torch.Tensor,
                   edge_weight: torch.Tensor = None):
        """Simple propagation"""
        row, col = edge_index
        deg = degree(row, embeddings.size(0), dtype=embeddings.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if edge_weight is None:
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        else:
            norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        out = torch.zeros_like(embeddings)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i], edge_index[1, i]
            out[dst] += norm[i] * embeddings[src]
        
        return out
    
    def forward(self, users: torch.Tensor, items: torch.Tensor,
                edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """Forward pass"""
        user_emb, item_emb = self.compute_graph_convolution(edge_index, edge_weight)
        
        user_emb_batch = user_emb[users]
        item_emb_batch = item_emb[items]
        
        scores = (user_emb_batch * item_emb_batch).sum(dim=1)
        
        return scores
    
    def get_all_embeddings(self, edge_index: torch.Tensor,
                          edge_weight: torch.Tensor = None):
        """모든 embedding 반환"""
        return self.compute_graph_convolution(edge_index, edge_weight)


if __name__ == "__main__":
    # 모델 테스트
    print("="*70)
    print("GNN 모델 테스트")
    print("="*70)
    
    # Dummy data
    n_users, n_items = 100, 50
    embedding_dim = 32
    
    # Dummy edge index (bipartite graph)
    edge_index = torch.tensor([
        [0, 0, 1, 1, 2],  # users
        [100, 101, 100, 102, 101]  # items (offset by n_users)
    ], dtype=torch.long)
    
    # Test LightGCN
    print("\n1. LightGCN Test")
    model1 = LightGCN(n_users=n_users, n_items=n_items, 
                      embedding_dim=embedding_dim, n_layers=3)
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    scores = model1(users, items, edge_index)
    print(f"   Output scores shape: {scores.shape}")
    print(f"   Sample scores: {scores[:3]}")
    
    # Test NGCF
    print("\n2. NGCF Test")
    model2 = NGCF(n_users=n_users, n_items=n_items,
                  embedding_dim=embedding_dim, n_layers=2)
    scores = model2(users, items, edge_index)
    print(f"   Output scores shape: {scores.shape}")
    print(f"   Sample scores: {scores[:3]}")
    
    # Test SimpleGCN
    print("\n3. SimpleGCN Test")
    model3 = SimpleGCN(n_users=n_users, n_items=n_items,
                       embedding_dim=embedding_dim, n_layers=2)
    scores = model3(users, items, edge_index)
    print(f"   Output scores shape: {scores.shape}")
    print(f"   Sample scores: {scores[:3]}")
    
    print("\n✅ 모델 테스트 완료")
