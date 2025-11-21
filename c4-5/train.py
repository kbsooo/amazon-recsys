"""
학습 파이프라인

GNN 모델 학습을 위한 Trainer 클래스
- BPR Loss
- Negative Sampling
- 다양한 시각화
- Early Stopping
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple
import time


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking Loss"""
    
    def __init__(self):
        super(BPRLoss, self).__init__()
        
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor):
        """
        Args:
            pos_scores: Positive item scores [batch_size]
            neg_scores: Negative item scores [batch_size]
        
        Returns:
            BPR loss
        """
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        return loss


class InteractionDataset(Dataset):
    """User-Item Interaction Dataset for BPR"""
    
    def __init__(self, train_df, n_items: int, n_negatives: int = 1):
        """
        Args:
            train_df: Training dataframe
            n_items: Total number of items
            n_negatives: Number of negative samples per positive sample
        """
        self.train_df = train_df
        self.n_items = n_items
        self.n_negatives = n_negatives
        
        # User-item interactions
        self.user_items = {}
        for _, row in train_df.iterrows():
            user = int(row['user_idx'])
            item = int(row['item_idx'])
            if user not in self.user_items:
                self.user_items[user] = set()
            self.user_items[user].add(item)
        
        # All users with interactions
        self.users = list(self.user_items.keys())
        
    def __len__(self):
        return len(self.train_df) * self.n_negatives
    
    def __getitem__(self, idx):
        # Sample a user
        user = self.users[idx % len(self.users)]
        
        # Sample a positive item
        pos_items = list(self.user_items[user])
        pos_item = np.random.choice(pos_items)
        
        # Sample a negative item
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.user_items[user]:
            neg_item = np.random.randint(0, self.n_items)
        
        return user, pos_item, neg_item


class GNNTrainer:
    """GNN 모델 학습 클래스"""
    
    def __init__(self, model, edge_index, edge_weight=None, device='cpu'):
        """
        Args:
            model: GNN 모델
            edge_index: Graph edge indices
            edge_weight: Edge weights (optional)
            device: 디바이스
        """
        self.model = model.to(device)
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device) if edge_weight is not None else None
        self.device = device
        
        self.history = {
            'train_loss': [],
            'val_recall': [],
            'val_ndcg': [],
            'val_precision': [],
            'epoch_times': []
        }
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """한 epoch 학습"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            users, pos_items, neg_items = batch
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)
            
            # Forward pass
            pos_scores = self.model(users, pos_items, self.edge_index, self.edge_weight)
            neg_scores = self.model(users, neg_items, self.edge_index, self.edge_weight)
            
            # Loss
            loss = criterion(pos_scores, neg_scores)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def fit(self, train_df, val_df, n_items,
            epochs=50, batch_size=1024, learning_rate=0.001,
            n_negatives=1, early_stopping_patience=5,
            eval_every=1, verbose=True):
        """
        모델 학습
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            n_items: Total number of items
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            n_negatives: Number of negative samples
            early_stopping_patience: Early stopping patience
            eval_every: Evaluate every N epochs
            verbose: Print progress
        """
        # Dataset and DataLoader
        train_dataset = InteractionDataset(train_df, n_items, n_negatives)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                  shuffle=True, num_workers=0)
        
        # Optimizer and Loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = BPRLoss()
        
        # Early stopping
        best_val_recall = 0
        patience_counter = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"학습 시작: {epochs} epochs")
            print(f"{'='*70}")
        
        from evaluate import GNNEvaluator
        evaluator = GNNEvaluator(self.model, self.edge_index, self.edge_weight, self.device)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Evaluate
            if (epoch + 1) % eval_every == 0:
                metrics = evaluator.evaluate(train_df, val_df, k=20, verbose=False)
                val_recall = metrics['recall@20']
                val_ndcg = metrics['ndcg@20']
                val_precision = metrics['precision@20']
                
                # Save history
                self.history['train_loss'].append(train_loss)
                self.history['val_recall'].append(val_recall)
                self.history['val_ndcg'].append(val_ndcg)
                self.history['val_precision'].append(val_precision)
                
                epoch_time = time.time() - start_time
                self.history['epoch_times'].append(epoch_time)
                
                if verbose:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Loss: {train_loss:.4f} | "
                          f"Recall@20: {val_recall:.4f} | "
                          f"NDCG@20: {val_ndcg:.4f} | "
                          f"Precision@20: {val_precision:.4f} | "
                          f"Time: {epoch_time:.2f}s")
                
                # Early stopping
                if val_recall > best_val_recall:
                    best_val_recall = val_recall
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        if verbose:
            print(f"{'='*70}")
            print(f"학습 완료! Best Recall@20: {best_val_recall:.4f}")
            print(f"{'='*70}\n")
    
    def plot_training_history(self, figsize=(15, 5)):
        """학습 곡선 시각화"""
        if not self.history['train_loss']:
            print("학습 기록이 없습니다.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('BPR Loss', fontsize=11)
        axes[0].set_title('학습 Loss 곡선', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Recall & NDCG
        axes[1].plot(epochs, self.history['val_recall'], 'g-', linewidth=2, label='Recall@20')
        axes[1].plot(epochs, self.history['val_ndcg'], 'r-', linewidth=2, label='NDCG@20')
        axes[1].set_xlabel('Epoch', fontsize=11)
        axes[1].set_ylabel('Score', fontsize=11)
        axes[1].set_title('Validation Metrics', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        # Precision
        axes[2].plot(epochs, self.history['val_precision'], 'purple', linewidth=2, label='Precision@20')
        axes[2].set_xlabel('Epoch', fontsize=11)
        axes[2].set_ylabel('Precision@20', fontsize=11)
        axes[2].set_title('Validation Precision', fontsize=12, fontweight='bold')
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comparison(self, other_trainers: Dict[str, 'GNNTrainer'], 
                       metric='val_recall', figsize=(12, 6)):
        """여러 모델 비교 시각화"""
        plt.figure(figsize=figsize)
        
        # Plot this model
        epochs = range(1, len(self.history[metric]) + 1)
        plt.plot(epochs, self.history[metric], linewidth=2, 
                label=f'Current Model', marker='o', markersize=4)
        
        # Plot other models
        for name, trainer in other_trainers.items():
            if trainer.history[metric]:
                epochs_other = range(1, len(trainer.history[metric]) + 1)
                plt.plot(epochs_other, trainer.history[metric], 
                        linewidth=2, label=name, marker='s', markersize=4)
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'모델 비교: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # 테스트는 통합 노트북에서 수행
    print("학습 파이프라인 모듈 로드 완료")
    print("사용 예시:")
    print("  trainer = GNNTrainer(model, edge_index, device='mps')")
    print("  trainer.fit(train_df, val_df, n_items, epochs=50)")
    print("  trainer.plot_training_history()")
