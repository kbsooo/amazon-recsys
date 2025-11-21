"""
평가 시스템

다양한 평가 지표:
- Recall@K
- NDCG@K  
- Precision@K
- MRR (Mean Reciprocal Rank)
- Hit Rate@K
- Coverage
- Diversity
- Novelty
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple
from tqdm import tqdm


class GNNEvaluator:
    """GNN 모델 평가 클래스"""
    
    def __init__(self, model, edge_index, edge_weight=None, device='cpu'):
        """
        Args:
            model: GNN 모델
            edge_index: Graph edge indices
            edge_weight: Edge weights (optional)
            device: 디바이스
        """
        self.model = model
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.device = device
        
    def get_user_item_scores(self, user_idx: int):
        """특정 사용자의 모든 아이템에 대한 점수 계산"""
        self.model.eval()
        
        with torch.no_grad():
            # Get all embeddings
            user_emb, item_emb = self.model.get_all_embeddings(
                self.edge_index, self.edge_weight
            )
            
            # User embedding
            user_emb_single = user_emb[user_idx].unsqueeze(0)  # [1, emb_dim]
            
            # Scores for all items
            scores = torch.matmul(user_emb_single, item_emb.t()).squeeze()  # [n_items]
            
        return scores.cpu().numpy()
    
    def recall_at_k(self, recommended_items: List[int], 
                    true_items: List[int], k: int) -> float:
        """
        Recall@K: 실제 관심 아이템 중 추천된 비율
        
        Args:
            recommended_items: 추천된 아이템 리스트 (top-k)
            true_items: 실제 관심 아이템 리스트
            k: K value
        
        Returns:
            Recall@K
        """
        if len(true_items) == 0:
            return 0.0
        
        recommended_set = set(recommended_items[:k])
        true_set = set(true_items)
        
        hits = len(recommended_set & true_set)
        return hits / len(true_set)
    
    def ndcg_at_k(self, recommended_items: List[int], 
                  true_items: List[int], k: int) -> float:
        """
        NDCG@K: Normalized Discounted Cumulative Gain
        
        순위를 고려한 평가 지표
        """
        if len(true_items) == 0:
            return 0.0
        
        true_set = set(true_items)
        dcg = 0.0
        
        for i, item in enumerate(recommended_items[:k]):
            if item in true_set:
                # i+1 because index starts at 0
                dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
        
        # Ideal DCG
        idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_set), k))])
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def precision_at_k(self, recommended_items: List[int],
                      true_items: List[int], k: int) -> float:
        """
        Precision@K: 추천된 아이템 중 실제 관심 아이템 비율
        """
        if k == 0:
            return 0.0
        
        recommended_set = set(recommended_items[:k])
        true_set = set(true_items)
        
        hits = len(recommended_set & true_set)
        return hits / k
    
    def mrr(self, recommended_items: List[int], true_items: List[int]) -> float:
        """
        MRR: Mean Reciprocal Rank
        
        첫 번째 관련 아이템의 순위의 역수
        """
        true_set = set(true_items)
        
        for i, item in enumerate(recommended_items):
            if item in true_set:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def hit_rate_at_k(self, recommended_items: List[int],
                     true_items: List[int], k: int) -> float:
        """
        Hit Rate@K: 추천 목록에 관련 아이템이 하나라도 있는지
        """
        recommended_set = set(recommended_items[:k])
        true_set = set(true_items)
        
        return 1.0 if len(recommended_set & true_set) > 0 else 0.0
    
    def evaluate(self, train_df, test_df, k=20, verbose=True) -> Dict:
        """
        전체 평가
        
        Args:
            train_df: Training dataframe (추천 시 제외할 아이템)
            test_df: Test dataframe (평가 대상)
            k: K value
            verbose: Print progress
        
        Returns:
            Dictionary of metrics
        """
        # Train set으로 user-item pairs 생성
        train_user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            train_user_items[int(row['user_idx'])].add(int(row['item_idx']))
        
        # Test set으로 user-item pairs 생성
        test_user_items = defaultdict(set)
        for _, row in test_df.iterrows():
            test_user_items[int(row['user_idx'])].add(int(row['item_idx']))
        
        # Metrics 초기화
        recalls = []
        ndcgs = []
        precisions = []
        mrrs = []
        hit_rates = []
        
        users_evaluated = list(test_user_items.keys())
        
        if verbose:
            users_evaluated = tqdm(users_evaluated, desc=f"Evaluating")
        
        for user_idx in users_evaluated:
            # Get scores for all items
            scores = self.get_user_item_scores(user_idx)
            
            # Exclude training items
            train_items = train_user_items.get(user_idx, set())
            for item in train_items:
                scores[item] = -np.inf
            
            # Get top-k recommendations
            top_k_items = np.argsort(scores)[::-1][:k].tolist()
            
            # True items
            true_items = list(test_user_items[user_idx])
            
            # Compute metrics
            recalls.append(self.recall_at_k(top_k_items, true_items, k))
            ndcgs.append(self.ndcg_at_k(top_k_items, true_items, k))
            precisions.append(self.precision_at_k(top_k_items, true_items, k))
            mrrs.append(self.mrr(top_k_items, true_items))
            hit_rates.append(self.hit_rate_at_k(top_k_items, true_items, k))
        
        # Average metrics
        metrics = {
            f'recall@{k}': np.mean(recalls),
            f'ndcg@{k}': np.mean(ndcgs),
            f'precision@{k}': np.mean(precisions),
            'mrr': np.mean(mrrs),
            f'hit_rate@{k}': np.mean(hit_rates),
            'n_users': len(users_evaluated)
        }
        
        return metrics
    
    def evaluate_by_user_group(self, train_df, test_df, k=20) -> Dict:
        """
        사용자 그룹별 평가 (Cold-start vs Regular users)
        
        Returns:
            Dictionary with metrics for each group
        """
        # User interaction counts
        user_counts = train_df.groupby('user_idx').size().to_dict()
        
        # Test set
        test_user_items = defaultdict(set)
        for _, row in test_df.iterrows():
            test_user_items[int(row['user_idx'])].add(int(row['item_idx']))
        
        # Train set
        train_user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            train_user_items[int(row['user_idx'])].add(int(row['item_idx']))
        
        # Split users by groups
        cold_start_users = []
        regular_users = []
        active_users = []
        
        for user_idx in test_user_items.keys():
            count = user_counts.get(user_idx, 0)
            if count <= 10:
                cold_start_users.append(user_idx)
            elif count <= 50:
                regular_users.append(user_idx)
            else:
                active_users.append(user_idx)
        
        results = {}
        
        # Evaluate each group
        for group_name, user_list in [('cold_start', cold_start_users),
                                       ('regular', regular_users),
                                       ('active', active_users)]:
            if len(user_list) == 0:
                continue
            
            recalls, ndcgs, precisions = [], [], []
            
            for user_idx in tqdm(user_list, desc=f"Evaluating {group_name}"):
                scores = self.get_user_item_scores(user_idx)
                
                # Exclude train items
                for item in train_user_items.get(user_idx, set()):
                    scores[item] = -np.inf
                
                top_k_items = np.argsort(scores)[::-1][:k].tolist()
                true_items = list(test_user_items[user_idx])
                
                recalls.append(self.recall_at_k(top_k_items, true_items, k))
                ndcgs.append(self.ndcg_at_k(top_k_items, true_items, k))
                precisions.append(self.precision_at_k(top_k_items, true_items, k))
            
            results[group_name] = {
                f'recall@{k}': np.mean(recalls),
                f'ndcg@{k}': np.mean(ndcgs),
                f'precision@{k}': np.mean(precisions),
                'n_users': len(user_list)
            }
        
        return results
    
    def compute_coverage(self, train_df, test_df, k=20) -> float:
        """
        Coverage: 전체 아이템 중 추천에 포함된 아이템 비율
        """
        recommended_items = set()
        
        test_users = test_df['user_idx'].unique()
        train_user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            train_user_items[int(row['user_idx'])].add(int(row['item_idx']))
        
        for user_idx in test_users:
            scores = self.get_user_item_scores(int(user_idx))
            
            # Exclude train items
            for item in train_user_items.get(int(user_idx), set()):
                scores[item] = -np.inf
            
            top_k_items = np.argsort(scores)[::-1][:k]
            recommended_items.update(top_k_items.tolist())
        
        n_items = self.model.n_items
        coverage = len(recommended_items) / n_items
        
        return coverage
    
    def plot_metrics_comparison(self, metrics_dict: Dict[str, Dict], 
                               figsize=(12, 6)):
        """
        여러 모델의 metrics 비교 시각화
        
        Args:
            metrics_dict: {model_name: metrics_dict}
        """
        # Extract metrics
        model_names = list(metrics_dict.keys())
        recall_values = [metrics_dict[name].get('recall@20', 0) for name in model_names]
        ndcg_values = [metrics_dict[name].get('ndcg@20', 0) for name in model_names]
        precision_values = [metrics_dict[name].get('precision@20', 0) for name in model_names]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        x = np.arange(len(model_names))
        width = 0.6
        
        # Recall
        axes[0].bar(x, recall_values, width, color='skyblue', edgecolor='black')
        axes[0].set_ylabel('Recall@20', fontsize=11)
        axes[0].set_title('Recall@20 비교', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # NDCG
        axes[1].bar(x, ndcg_values, width, color='lightcoral', edgecolor='black')
        axes[1].set_ylabel('NDCG@20', fontsize=11)
        axes[1].set_title('NDCG@20 비교', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Precision
        axes[2].bar(x, precision_values, width, color='lightgreen', edgecolor='black')
        axes[2].set_ylabel('Precision@20', fontsize=11)
        axes[2].set_title('Precision@20 비교', fontsize=12, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_user_group_performance(self, group_results: Dict, 
                                    figsize=(12, 5)):
        """사용자 그룹별 성능 시각화"""
        groups = list(group_results.keys())
        metrics = ['recall@20', 'ndcg@20', 'precision@20']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        
        for idx, metric in enumerate(metrics):
            values = [group_results[g].get(metric, 0) for g in groups]
            
            axes[idx].bar(range(len(groups)), values, color=['coral', 'skyblue', 'lightgreen'],
                         edgecolor='black', alpha=0.7)
            axes[idx].set_xticks(range(len(groups)))
            axes[idx].set_xticklabels(groups, rotation=45, ha='right')
            axes[idx].set_ylabel(metric.replace('@', ' @'), fontsize=11)
            axes[idx].set_title(f'{metric.upper()} by User Group', 
                               fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("평가 시스템 모듈 로드 완료")
    print("사용 예시:")
    print("  evaluator = GNNEvaluator(model, edge_index, device='mps')")
    print("  metrics = evaluator.evaluate(train_df, test_df, k=20)")
    print("  group_metrics = evaluator.evaluate_by_user_group(train_df, test_df)")
