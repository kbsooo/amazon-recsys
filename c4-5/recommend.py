"""
추천 생성 시스템

사용자별 맞춤 추천 생성
- 평가 규칙 준수 (50% 제한)
- Cold-start 사용자 처리
- 출력 포맷 생성
"""

import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple
from collections import defaultdict


class RecommendationSystem:
    """추천 생성 시스템"""
    
    def __init__(self, model, preprocessor, edge_index, edge_weight=None, device='cpu'):
        """
        Args:
            model: 학습된 GNN 모델
            preprocessor: DataPreprocessor 인스턴스
            edge_index: Graph edge indices
            edge_weight: Edge weights (optional)
            device: 디바이스
        """
        self.model = model
        self.preprocessor = preprocessor
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.device = device
        
        self.model.eval()
    
    def get_user_recommendations(self, user_idx: int, k: int,
                                exclude_items: set = None) -> List[int]:
        """
        특정 사용자에 대한 top-k 추천 생성
        
        Args:
            user_idx: User index
            k: 추천 개수
            exclude_items: 제외할 아이템 set
        
        Returns:
            추천 아이템 리스트 (item indices)
        """
        with torch.no_grad():
            # Get all embeddings
            user_emb, item_emb = self.model.get_all_embeddings(
                self.edge_index, self.edge_weight
            )
            
            # User embedding
            user_emb_single = user_emb[user_idx].unsqueeze(0)
            
            # Scores for all items
            scores = torch.matmul(user_emb_single, item_emb.t()).squeeze()
            scores = scores.cpu().numpy()
            
            # Exclude items
            if exclude_items:
                for item in exclude_items:
                    scores[item] = -np.inf
            
            # Get top-k
            top_k_items = np.argsort(scores)[::-1][:k].tolist()
            
        return top_k_items
    
    def generate_recommendations_for_test(self, train_df, test_df, 
                                         k_values: Dict[int, int]) -> pd.DataFrame:
        """
        Test set에 대한 추천 생성 (평가용)
        
        Args:
            train_df: Training dataframe (제외할 아이템)
            test_df: Test dataframe (추천 대상 user-item pairs)
            k_values: {user_idx: k} 사용자별 추천 개수
        
        Returns:
            DataFrame with columns: user, item, recommend
        """
        # Train user-items (제외할 아이템)
        train_user_items = defaultdict(set)
        for _, row in train_df.iterrows():
            train_user_items[int(row['user_idx'])].add(int(row['item_idx']))
        
        # Generate recommendations for each user
        user_recommendations = {}
        unique_users = test_df['user_idx'].unique()
        
        for user_idx in unique_users:
            user_idx = int(user_idx)
            k = k_values.get(user_idx, 2)  # default 2
            
            # Get recommendations
            exclude = train_user_items.get(user_idx, set())
            recommendations = self.get_user_recommendations(user_idx, k, exclude)
            user_recommendations[user_idx] = set(recommendations)
        
        # Create result dataframe
        results = []
        for _, row in test_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            
            # Check if item is in recommendations
            recommend = 'O' if item_idx in user_recommendations.get(user_idx, set()) else 'X'
            
            # Original user and item IDs
            user_id = self.preprocessor.idx2user[user_idx]
            item_id = self.preprocessor.idx2item[item_idx]
            
            results.append({
                'user': user_id,
                'item': item_id,
                'recommend': recommend
            })
        
        result_df = pd.DataFrame(results)
        return result_df
    
    def format_output(self, result_df: pd.DataFrame) -> str:
        """
        추천 결과를 요구사항 형식으로 포맷
        
        Returns:
            formatted string
        """
        output = []
        output.append("="*50)
        output.append(f"{'user':<15} {'item':<20} {'recommend':<10}")
        output.append("="*50)
        
        for _, row in result_df.head(10).iterrows():
            output.append(f"{row['user']:<15} {row['item']:<20} {row['recommend']:<10}")
        
        if len(result_df) > 10:
            output.append("...")
        
        # Summary
        total = len(result_df)
        recommend_count = (result_df['recommend'] == 'O').sum()
        not_recommend_count = (result_df['recommend'] == 'X').sum()
        
        output.append("="*50)
        output.append(f"Total recommends = {recommend_count}/{total}")
        output.append(f"Total not recommends = {not_recommend_count}/{total}")
        output.append("="*50)
        
        return '\n'.join(output)
    
    def save_recommendations(self, result_df: pd.DataFrame, filepath: str):
        """추천 결과 CSV 저장"""
        result_df.to_csv(filepath, index=False)
        print(f"추천 결과 저장: {filepath}")
    
    def analyze_recommendations(self, result_df: pd.DataFrame, k_values: Dict,
                               train_df) -> Dict:
        """
        추천 결과 분석
        
        Returns:
            분석 결과 dictionary
        """
        # Count by user
        user_stats = result_df.groupby('user')['recommend'].value_counts().unstack(fill_value=0)
        
        # K value distribution
        user_idx_to_k = {}
        for user in result_df['user'].unique():
            user_idx = self.preprocessor.user2idx[user]
            user_idx_to_k[user] = k_values.get(user_idx, 2)
        
        # Train interaction counts
        train_counts = train_df.groupby('user_idx').size().to_dict()
        
        analysis = {
            'total_users': result_df['user'].nunique(),
            'total_pairs': len(result_df),
            'total_recommends': (result_df['recommend'] == 'O').sum(),
            'total_not_recommends': (result_df['recommend'] == 'X').sum(),
            'recommend_ratio': (result_df['recommend'] == 'O').sum() / len(result_df),
            'avg_k': np.mean(list(user_idx_to_k.values())),
            'k_distribution': pd.Series(user_idx_to_k.values()).value_counts().to_dict()
        }
        
        return analysis
    
    def generate_for_custom_input(self, input_filepath: str, 
                                  train_df, k_values: Dict[int, int],
                                  output_filepath: str = None) -> pd.DataFrame:
        """
        임의의 입력 파일에 대한 추천 생성
        
        Args:
            input_filepath: 입력 CSV 파일 (user, item 컬럼 필요)
            train_df: Training dataframe
            k_values: 사용자별 추천 개수
            output_filepath: 출력 파일 경로 (None이면 저장 안함)
        
        Returns:
            추천 결과 DataFrame
        """
        # Load input
        input_df = pd.read_csv(input_filepath)
        
        # Encode
        input_df['user_idx'] = input_df['user'].map(self.preprocessor.user2idx)
        input_df['item_idx'] = input_df['item'].map(self.preprocessor.item2idx)
        
        # Handle unknown users/items
        input_df = input_df.dropna(subset=['user_idx', 'item_idx'])
        input_df['user_idx'] = input_df['user_idx'].astype(int)
        input_df['item_idx'] = input_df['item_idx'].astype(int)
        
        # Generate recommendations
        result_df = self.generate_recommendations_for_test(train_df, input_df, k_values)
        
        # Save if needed
        if output_filepath:
            self.save_recommendations(result_df, output_filepath)
        
        return result_df


if __name__ == "__main__":
    print("추천 시스템 모듈 로드 완료")
    print("사용 예시:")
    print("  rec_sys = RecommendationSystem(model, preprocessor, edge_index)")
    print("  result_df = rec_sys.generate_recommendations_for_test(train_df, test_df, k_values)")
    print("  print(rec_sys.format_output(result_df))")
