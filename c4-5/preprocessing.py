"""
데이터 전처리 유틸리티

Amazon 추천 시스템 데이터 전처리 함수들
- User/Item ID 인코딩
- Train/Validation/Test 분할
- 추천 개수 계산
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Tuple, Dict


class DataPreprocessor:
    """데이터 전처리 클래스"""
    
    def __init__(self, good_rating_threshold: float = 4.0, 
                 cold_start_threshold: int = 10,
                 recommend_ratio: float = 0.5,
                 min_recommend: int = 2):
        """
        Args:
            good_rating_threshold: Good rating 기준 (default: 4.0)
            cold_start_threshold: Cold-start 기준 interaction 수 (default: 10)
            recommend_ratio: 일반 사용자 추천 비율 (default: 0.5 = 50%)
            min_recommend: Cold-start 사용자 최소 추천 개수 (default: 2)
        """
        self.good_rating_threshold = good_rating_threshold
        self.cold_start_threshold = cold_start_threshold
        self.recommend_ratio = recommend_ratio
        self.min_recommend = min_recommend
        
        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}
        self.n_users = 0
        self.n_items = 0
    
    def fit(self, df: pd.DataFrame) -> 'DataPreprocessor':
        """
        User/Item ID 인코딩 매핑 생성
        
        Args:
            df: Raw dataframe
            
        Returns:
            self
        """
        # User encoding
        unique_users = df['user'].unique()
        self.user2idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx2user = {idx: user for user, idx in self.user2idx.items()}
        self.n_users = len(self.user2idx)
        
        # Item encoding
        unique_items = df['item'].unique()
        self.item2idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx2item = {idx: item for item, idx in self.item2idx.items()}
        self.n_items = len(self.item2idx)
        
        print(f"인코딩 완료: {self.n_users:,} users, {self.n_items:,} items")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrame에 인코딩 적용
        
        Args:
            df: Raw dataframe
            
        Returns:
            Encoded dataframe with user_idx, item_idx columns
        """
        df = df.copy()
        df['user_idx'] = df['user'].map(self.user2idx)
        df['item_idx'] = df['item'].map(self.item2idx)
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step"""
        return self.fit(df).transform(df)
    
    def split_train_val_test(self, df: pd.DataFrame, 
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Train/Validation/Test 분할
        사용자별로 good rating (≥threshold)만 사용하여 시간순 분할
        
        Args:
            df: Encoded dataframe
            train_ratio: Train 비율
            val_ratio: Validation 비율
            test_ratio: Test 비율
            random_state: Random seed
        
        Returns:
            (train_df, val_df, test_df)
        """
        np.random.seed(random_state)
        
        # Good rating만 필터링
        good_df = df[df['rating'] >= self.good_rating_threshold].copy()
        
        train_data = []
        val_data = []
        test_data = []
        
        # 사용자별로 분할
        for user_idx in range(self.n_users):
            user_df = good_df[good_df['user_idx'] == user_idx]
            
            if len(user_df) == 0:
                continue
            
            # Shuffle (시간정보 없으므로 랜덤)
            user_df = user_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
            
            n_good = len(user_df)
            
            if n_good >= 3:
                # 최소 3개 이상일 때만 3-way split
                tr_end = max(1, int(train_ratio * n_good))
                val_end = max(tr_end + 1, tr_end + int(val_ratio * n_good))
                
                train_data.append(user_df.iloc[:tr_end])
                val_data.append(user_df.iloc[tr_end:val_end])
                test_data.append(user_df.iloc[val_end:])
            elif n_good == 2:
                # 2개면 train 1, val 0, test 1
                train_data.append(user_df.iloc[:1])
                test_data.append(user_df.iloc[1:])
            else:  # n_good == 1
                # 1개면 모두 train
                train_data.append(user_df)
        
        train_df = pd.concat(train_data, ignore_index=True)
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        print(f"데이터 분할 완료:")
        print(f"  Train: {len(train_df):,} ({len(train_df)/len(good_df)*100:.1f}%)")
        print(f"  Val:   {len(val_df):,} ({len(val_df)/len(good_df)*100:.1f}%)")
        print(f"  Test:  {len(test_df):,} ({len(test_df)/len(good_df)*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def get_k_for_user(self, user_idx: int, user_interaction_counts: Dict[int, int]) -> int:
        """
        사용자별 추천 개수 계산
        
        Args:
            user_idx: User index
            user_interaction_counts: Dict mapping user_idx to interaction count
        
        Returns:
            추천 개수 k
        """
        count = user_interaction_counts.get(user_idx, 0)
        
        if count <= self.cold_start_threshold:
            return self.min_recommend
        else:
            return int(count * self.recommend_ratio)
    
    def compute_user_k_values(self, train_df: pd.DataFrame) -> Dict[int, int]:
        """
        모든 사용자의 k값 (추천 개수) 계산
        
        Args:
            train_df: Training dataframe
        
        Returns:
            Dict mapping user_idx to k value
        """
        # Good rating만 카운트
        good_train = train_df[train_df['rating'] >= self.good_rating_threshold]
        user_counts = good_train.groupby('user_idx').size().to_dict()
        
        k_values = {}
        for user_idx in range(self.n_users):
            k_values[user_idx] = self.get_k_for_user(user_idx, user_counts)
        
        return k_values


def load_and_preprocess_data(data_path: str,
                             good_rating_threshold: float = 4.0,
                             cold_start_threshold: int = 10,
                             recommend_ratio: float = 0.5,
                             min_recommend: int = 2,
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             random_state: int = 42) -> Tuple:
    """
    데이터 로딩부터 전처리까지 전체 파이프라인
    
    Args:
        data_path: CSV 파일 경로
        good_rating_threshold: Good rating 기준
        cold_start_threshold: Cold-start 기준
        recommend_ratio: 추천 비율
        min_recommend: 최소 추천 개수
        train_ratio: Train 비율
        val_ratio: Validation 비율
        test_ratio: Test 비율
        random_state: Random seed
    
    Returns:
        (preprocessor, train_df, val_df, test_df, k_values)
    """
    # 1. 데이터 로드
    print(f"데이터 로딩: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  {len(df):,} interactions loaded")
    
    # 2. Preprocessor 초기화 및 fit
    preprocessor = DataPreprocessor(
        good_rating_threshold=good_rating_threshold,
        cold_start_threshold=cold_start_threshold,
        recommend_ratio=recommend_ratio,
        min_recommend=min_recommend
    )
    
    # 3. 인코딩
    df_encoded = preprocessor.fit_transform(df)
    
    # 4. Train/Val/Test 분할
    train_df, val_df, test_df = preprocessor.split_train_val_test(
        df_encoded, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # 5. K values 계산
    k_values = preprocessor.compute_user_k_values(train_df)
    print(f"  K values computed: 평균 {np.mean(list(k_values.values())):.2f}")
    
    return preprocessor, train_df, val_df, test_df, k_values


def create_edge_set(df: pd.DataFrame) -> set:
    """
    DataFrame에서 (user_idx, item_idx) edge set 생성
    
    Args:
        df: DataFrame with user_idx and item_idx
        
    Returns:
        Set of (user_idx, item_idx) tuples
    """
    edges = set()
    for _, row in df.iterrows():
        edges.add((int(row['user_idx']), int(row['item_idx'])))
    return edges


if __name__ == "__main__":
    # 테스트
    import sys
    
    data_path = "../data/amazon_train.csv"
    
    print("="*70)
    print("데이터 전처리 테스트")
    print("="*70)
    
    preprocessor, train_df, val_df, test_df, k_values = load_and_preprocess_data(
        data_path=data_path,
        good_rating_threshold=4.0,
        cold_start_threshold=10,
        recommend_ratio=0.5,
        min_recommend=2,
        random_state=42
    )
    
    print("\n" + "="*70)
    print("결과 확인")
    print("="*70)
    print(f"Users: {preprocessor.n_users:,}")
    print(f"Items: {preprocessor.n_items:,}")
    print(f"\nTrain shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")
    print(f"\nK values stats:")
    k_vals = list(k_values.values())
    print(f"  Min: {min(k_vals)}")
    print(f"  Max: {max(k_vals)}")
    print(f"  Mean: {np.mean(k_vals):.2f}")
    print(f"  Median: {np.median(k_vals):.2f}")
    
    # Edge set 생성 테스트
    train_edges = create_edge_set(train_df)
    val_edges = create_edge_set(val_df)
    test_edges = create_edge_set(test_df)
    
    print(f"\nEdge sets:")
    print(f"  Train edges: {len(train_edges):,}")
    print(f"  Val edges:   {len(val_edges):,}")
    print(f"  Test edges:  {len(test_edges):,}")
    
    # Overlap 확인
    train_val_overlap = len(train_edges & val_edges)
    train_test_overlap = len(train_edges & test_edges)
    val_test_overlap = len(val_edges & test_edges)
    
    print(f"\nOverlap (should be 0):")
    print(f"  Train-Val:  {train_val_overlap}")
    print(f"  Train-Test: {train_test_overlap}")
    print(f"  Val-Test:   {val_test_overlap}")
    
    print("\n✅ 전처리 테스트 완료")
