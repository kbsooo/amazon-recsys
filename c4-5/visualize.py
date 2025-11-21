"""
시각화 유틸리티

다양한 시각화 함수들:
- 학습 곡선
- 모델 비교
- 사용자 그룹별 성능
- Embedding 시각화 (t-SNE)
- Top-K 아이템 분석
- K값 분포
- Coverage & Diversity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from collections import Counter


class VisualizationUtils:
    """시각화 유틸리티 클래스"""
    
    @staticmethod
    def plot_data_overview(train_df, val_df, test_df, figsize=(15, 5)):
        """데이터셋 개요 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 데이터셋 크기
        datasets = ['Train', 'Validation', 'Test']
        sizes = [len(train_df), len(val_df), len(test_df)]
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        
        axes[0].bar(datasets, sizes, color=colors, edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Number of Interactions', fontsize=11)
        axes[0].set_title('데이터셋 크기', fontsize=12, fontweight='bold')
        axes[0].ticklabel_format(style='plain', axis='y')
        for i, v in enumerate(sizes):
            axes[0].text(i, v + 1000, f'{v:,}', ha='center', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Rating 분포 (Train만)
        rating_counts = train_df['rating'].value_counts().sort_index()
        axes[1].bar(rating_counts.index, rating_counts.values, 
                   color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Rating', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Rating 분포 (Train)', fontsize=12, fontweight='bold')
        axes[1].ticklabel_format(style='plain', axis='y')
        axes[1].grid(axis='y', alpha=0.3)
        
        # User interaction 분포
        user_counts = train_df.groupby('user_idx').size()
        axes[2].hist(user_counts, bins=50, color='darkorange', 
                    edgecolor='black', alpha=0.7)
        axes[2].set_xlabel('Number of Interactions per User', fontsize=11)
        axes[2].set_ylabel('Number of Users', fontsize=11)
        axes[2].set_title('사용자별 Interaction 분포', fontsize=12, fontweight='bold')
        axes[2].set_yscale('log')
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_k_distribution(k_values: Dict[int, int], figsize=(12, 5)):
        """K값 (추천 개수) 분포 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        k_list = list(k_values.values())
        
        # Histogram
        axes[0].hist(k_list, bins=30, color='mediumpurple', 
                    edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('K (Number of Recommendations)', fontsize=11)
        axes[0].set_ylabel('Number of Users', fontsize=11)
        axes[0].set_title('K값 분포 (Histogram)', fontsize=12, fontweight='bold')
        axes[0].axvline(np.mean(k_list), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(k_list):.1f}')
        axes[0].axvline(np.median(k_list), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(k_list):.1f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(k_list, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2))
        axes[1].set_ylabel('K (Number of Recommendations)', fontsize=11)
        axes[1].set_title('K값 분포 (Box Plot)', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Statistics
        print("="*50)
        print("K값 통계")
        print("="*50)
        print(f"평균: {np.mean(k_list):.2f}")
        print(f"중앙값: {np.median(k_list):.2f}")
        print(f"최소: {np.min(k_list)}")
        print(f"최대: {np.max(k_list)}")
        print(f"표준편차: {np.std(k_list):.2f}")
        print(f"Cold-start users (k=2): {sum(1 for k in k_list if k == 2):,} "
              f"({sum(1 for k in k_list if k == 2)/len(k_list)*100:.1f}%)")
    
    @staticmethod
    def plot_training_comparison(trainers: Dict[str, any], 
                                 metric='val_recall', figsize=(12, 6)):
        """여러 모델의 학습 곡선 비교"""
        plt.figure(figsize=figsize)
        
        for name, trainer in trainers.items():
            if hasattr(trainer, 'history') and trainer.history.get(metric):
                epochs = range(1, len(trainer.history[metric]) + 1)
                plt.plot(epochs, trainer.history[metric], 
                        linewidth=2.5, label=name, marker='o', markersize=5)
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        plt.title(f'모델 학습 곡선 비교: {metric.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_final_metrics_comparison(metrics_dict: Dict[str, Dict], 
                                      figsize=(14, 5)):
        """최종 모델 성능 비교 (막대 그래프)"""
        models = list(metrics_dict.keys())
        
        # 지표 추출
        recall_20 = [metrics_dict[m].get('recall@20', 0) for m in models]
        ndcg_20 = [metrics_dict[m].get('ndcg@20', 0) for m in models]
        precision_20 = [metrics_dict[m].get('precision@20', 0) for m in models]
        mrr = [metrics_dict[m].get('mrr', 0) for m in models]
        
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        x = np.arange(len(models))
        width = 0.6
        
        # Recall@20
        axes[0].bar(x, recall_20, width, color='skyblue', edgecolor='black', alpha=0.8)
        axes[0].set_ylabel('Recall@20', fontsize=11, fontweight='bold')
        axes[0].set_title('Recall@20', fontsize=12, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(recall_20):
            axes[0].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
        
        # NDCG@20
        axes[1].bar(x, ndcg_20, width, color='lightcoral', edgecolor='black', alpha=0.8)
        axes[1].set_ylabel('NDCG@20', fontsize=11, fontweight='bold')
        axes[1].set_title('NDCG@20', fontsize=12, fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(ndcg_20):
            axes[1].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
        
        # Precision@20
        axes[2].bar(x, precision_20, width, color='lightgreen', edgecolor='black', alpha=0.8)
        axes[2].set_ylabel('Precision@20', fontsize=11, fontweight='bold')
        axes[2].set_title('Precision@20', fontsize=12, fontweight='bold')
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        for i, v in enumerate(precision_20):
            axes[2].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
        
        # MRR
        axes[3].bar(x, mrr, width, color='gold', edgecolor='black', alpha=0.8)
        axes[3].set_ylabel('MRR', fontsize=11, fontweight='bold')
        axes[3].set_title('Mean Reciprocal Rank', fontsize=12, fontweight='bold')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(models, rotation=45, ha='right')
        axes[3].grid(axis='y', alpha=0.3)
        for i, v in enumerate(mrr):
            axes[3].text(i, v + 0.001, f'{v:.4f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_recommendation_analysis(result_df, figsize=(14, 5)):
        """추천 결과 분석 시각화"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Recommend vs Not Recommend
        recommend_counts = result_df['recommend'].value_counts()
        colors_pie = ['lightgreen', 'lightcoral']
        axes[0].pie(recommend_counts, labels=recommend_counts.index, 
                   autopct='%1.1f%%', colors=colors_pie, startangle=90,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[0].set_title('추천 비율', fontsize=12, fontweight='bold')
        
        # 2. User별 추천 개수 분포
        user_recommend_counts = result_df[result_df['recommend'] == 'O'].groupby('user').size()
        axes[1].hist(user_recommend_counts, bins=20, color='steelblue', 
                    edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('# of Recommendations per User', fontsize=11)
        axes[1].set_ylabel('# of Users', fontsize=11)
        axes[1].set_title('사용자별 추천 개수 분포', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        # 3. Item별 추천 횟수 Top 20
        item_recommend_counts = result_df[result_df['recommend'] == 'O']['item'].value_counts().head(20)
        axes[2].barh(range(len(item_recommend_counts)), item_recommend_counts.values, 
                    color='orange', edgecolor='black', alpha=0.7)
        axes[2].set_yticks(range(len(item_recommend_counts)))
        axes[2].set_yticklabels([f'...{item[-8:]}' for item in item_recommend_counts.index], 
                               fontsize=8)
        axes[2].set_xlabel('# of Recommendations', fontsize=11)
        axes[2].set_title('Top 20 추천된 아이템', fontsize=12, fontweight='bold')
        axes[2].invert_yaxis()
        axes[2].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_embeddings_tsne(model, edge_index, edge_weight=None, 
                            n_samples=1000, figsize=(12, 5)):
        """
        Embedding 시각화 (t-SNE)
        
        Note: sklearn 필요
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            print("sklearn이 설치되지 않았습니다. t-SNE 시각화를 건너뜁니다.")
            return
        
        import torch
        
        model.eval()
        with torch.no_grad():
            user_emb, item_emb = model.get_all_embeddings(edge_index, edge_weight)
        
        user_emb_np = user_emb.cpu().numpy()
        item_emb_np = item_emb.cpu().numpy()
        
        # 샘플링
        n_users = min(n_samples, user_emb_np.shape[0])
        n_items = min(n_samples, item_emb_np.shape[0])
        
        user_sample = user_emb_np[np.random.choice(user_emb_np.shape[0], n_users, replace=False)]
        item_sample = item_emb_np[np.random.choice(item_emb_np.shape[0], n_items, replace=False)]
        
        # t-SNE
        print(f"t-SNE 수행 중... (샘플: {n_users} users, {n_items} items)")
        embeddings = np.vstack([user_sample, item_sample])
        labels = ['User'] * n_users + ['Item'] * n_items
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 시각화
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: User vs Item
        user_idx = np.array(labels) == 'User'
        item_idx = np.array(labels) == 'Item'
        
        axes[0].scatter(embeddings_2d[user_idx, 0], embeddings_2d[user_idx, 1],
                       c='skyblue', label='User', alpha=0.6, s=20)
        axes[0].scatter(embeddings_2d[item_idx, 0], embeddings_2d[item_idx, 1],
                       c='coral', label='Item', alpha=0.6, s=20)
        axes[0].set_title('t-SNE: User vs Item Embeddings', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Plot 2: Combined
        axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                       c=['skyblue' if l == 'User' else 'coral' for l in labels],
                       alpha=0.5, s=15)
        axes[1].set_title('t-SNE: All Embeddings', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print("t-SNE 시각화 완료!")


if __name__ == "__main__":
    print("시각화 유틸리티 모듈 로드 완료")
    print("사용 예시:")
    print("  viz = VisualizationUtils()")
    print("  viz.plot_data_overview(train_df, val_df, test_df)")
    print("  viz.plot_k_distribution(k_values)")
