#%%
"""
SimGCL 기반 추론 시스템
최종 추천 결과를 생성하는 inference 모듈
"""
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import sys

# 모듈 경로 추가
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from model import LightGCN_SimGCL

class SimGCLInference:
    def __init__(self, model_path, data_dir):
        """
        SimGCL 추론 시스템 초기화
        
        Args:
            model_path: SimGCL 모델 경로
            data_dir: 데이터 디렉토리 경로
        """
        # Device 설정
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        print(f"Device: {self.device}")
        
        # 모델 및 데이터 로드
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        # ID 매핑 로드
        with open(data_dir / 'user2idx.pkl', 'rb') as f:
            self.user2idx = pickle.load(f)
        with open(data_dir / 'item2idx.pkl', 'rb') as f:
            self.item2idx = pickle.load(f)
        with open(data_dir / 'idx2user.pkl', 'rb') as f:
            self.idx2user = pickle.load(f)
        with open(data_dir / 'idx2item.pkl', 'rb') as f:
            self.idx2item = pickle.load(f)
        
        # User K 정보 로드
        with open(data_dir / 'user_k.pkl', 'rb') as f:
            self.user_k = pickle.load(f)
        
        # User Train Items 로드 (추천에서 제외하기 위함)
        with open(data_dir / 'user_train_items.pkl', 'rb') as f:
            self.user_train_items = pickle.load(f)
        
        # 그래프 데이터 로드
        graph_data = torch.load(data_dir / 'train_graph.pt', map_location=self.device, weights_only=False)
        self.edge_index = graph_data['edge_index'].to(self.device)
        self.edge_weight = graph_data['cca_weight'].to(self.device)  # SimGCL은 CCA weight 사용
        
        # 모델 초기화 및 로드
        self.model = LightGCN_SimGCL(
            config['n_users'],
            config['n_items'],
            config['emb_dim'],
            config['n_layers'],
            eps=config['eps']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        print("✅ SimGCL Inference 시스템 초기화 완료")
        
        # 임베딩 미리 계산 (속도 향상)
        with torch.no_grad():
            self.user_emb, self.item_emb = self.model(self.edge_index, self.edge_weight, perturbed=False)
    
    def predict(self, test_df):
        """
        테스트 데이터에 대한 추천 결과 생성
        
        Args:
            test_df: 테스트 데이터프레임 (columns: user, item)
        
        Returns:
            추천 결과 데이터프레임 (columns: user, item, recommend)
        """
        results = []
        
        for user_id, group in test_df.groupby('user'):
            # 새로운 유저인 경우 모두 X
            if user_id not in self.user2idx:
                for _, row in group.iterrows():
                    results.append({
                        'user': row['user'],
                        'item': row['item'],
                        'recommend': 'X'
                    })
                continue
            
            u_idx = self.user2idx[user_id]
            
            # 이 유저의 K값
            K = self.user_k.get(u_idx, 2)
            MIN_K = 2  # 최소 추천 개수
            
            # 아이템별 점수 계산
            items_to_score = []
            valid_rows = []
            
            for _, row in group.iterrows():
                item_id = row['item']
                
                # 새로운 아이템이거나 이미 train set에 있는 아이템은 제외
                if item_id not in self.item2idx:
                    results.append({
                        'user': row['user'],
                        'item': row['item'],
                        'recommend': 'X'
                    })
                    continue
                
                i_idx = self.item2idx[item_id]
                
                # Train set에 이미 있는 아이템은 제외
                if i_idx in self.user_train_items.get(u_idx, set()):
                    results.append({
                        'user': row['user'],
                        'item': row['item'],
                        'recommend': 'X'
                    })
                    continue
                
                items_to_score.append((i_idx, row))
                valid_rows.append(row)
            
            if not items_to_score:
                continue
            
            # 배치 점수 계산
            item_indices = torch.LongTensor([i for i, _ in items_to_score]).to(self.device)
            
            with torch.no_grad():
                scores = (self.user_emb[u_idx] * self.item_emb[item_indices]).sum(dim=1).cpu().numpy()
            
            # Top-K 선정 (최소 MIN_K, 최대 K)
            num_recommend = max(MIN_K, min(K, len(scores) // 2))  # README의 50% 규칙
            
            # 점수 기준 정렬
            top_indices = np.argsort(scores)[-num_recommend:]
            top_set = set(top_indices)
            
            # 결과 생성
            for idx, (item_idx, row) in enumerate(items_to_score):
                recommend = 'O' if idx in top_set else 'X'
                results.append({
                    'user': row['user'],
                    'item': row['item'],
                    'recommend': recommend
                })
        
        return pd.DataFrame(results)

def print_formatted_results(preds_df):
    """
    README.md 요구사항에 맞춘 출력 포맷팅 함수
    """
    total_cnt = len(preds_df)
    rec_cnt = len(preds_df[preds_df['recommend'] == 'O'])
    not_rec_cnt = total_cnt - rec_cnt
    
    print("="*20)
    print(f"{'user':<7}{'item':<7}{'recommend':<9}")
    
    for _, row in preds_df.iterrows():
        print(f"{str(row['user']):<7}{str(row['item']):<7}{row['recommend']:<9}")
    
    print("="*20)
    print(f"Total recommends: {rec_cnt}/{total_cnt}")
    print(f"Not recommend: {not_rec_cnt}/{total_cnt}")

# 테스트 실행
if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / 'data'
    models_dir = Path(__file__).parent.parent / 'models'
    
    # Inference 시스템 초기화
    inference = SimGCLInference(
        model_path=models_dir / 'simgcl_final.pt',
        data_dir=data_dir
    )
    
    # 샘플 테스트
    test_file = data_dir / 'test_split.csv'
    if test_file.exists():
        test_df = pd.read_csv(test_file)
        
        # 처음 100명의 유저만 테스트 (빠른 실행)
        sample_users = test_df['user_idx'].unique()[:100]
        sample_df = test_df[test_df['user_idx'].isin(sample_users)][['user', 'item']]
        
        print("\n추론 시작...")
        predictions = inference.predict(sample_df)
        
        print("\n샘플 결과:")
        print_formatted_results(predictions.head(20))
        
        print(f"\n✅ 추론 완료. 총 {len(predictions)}건 처리")
    else:
        print("테스트 파일이 없습니다.")
