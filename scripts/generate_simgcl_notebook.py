import json
import os

def read_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def create_simgcl_notebook():
    # 1. Read source files
    try:
        model_code = read_file('src/model.py')
        train_code = read_file('scripts/04_train_simgcl.py')
    except FileNotFoundError:
        # Fallback if running from scripts/
        model_code = read_file('../src/model.py')
        train_code = read_file('04_train_simgcl.py')

    # 2. Process Model Code
    # Extract only SimGCL related code
    model_lines = model_code.split('\n')
    processed_model_lines = []
    
    # LightGCN_SimGCL 클래스와 InfoNCE Loss만 추출
    include = False
    for line in model_lines:
        # SimGCL 섹션 시작
        if 'LightGCN + SimGCL' in line or 'class LightGCN_SimGCL' in line:
            include = True
        
        # 테스트 섹션 시작하면 종료
        if 'if __name__ == "__main__":' in line:
            break
            
        # InfoNCE Loss 함수 포함
        if 'def compute_infonce_loss' in line:
            include = True
        
        if include:
            processed_model_lines.append(line)
            
        # InfoNCE 함수 끝
        if include and line.strip().startswith('return loss'):
            # 몇 줄 더 추가
            continue
    
    model_source = '\n'.join(processed_model_lines)

    # 3. Process Train Code
    # Replace paths
    train_code = train_code.replace('../data/', '/kaggle/input/amazon/')
    train_code = train_code.replace('../models/', 'models/')
    train_code = train_code.replace('../outputs/', 'outputs/')
    
    # Remove sys.path and import model
    train_lines = train_code.split('\n')
    processed_train_lines = []
    for line in train_lines:
        if "sys.path.append" in line or "from model import" in line:
            continue
        processed_train_lines.append(line)
    train_source = '\n'.join(processed_train_lines)

    # 4. Create Notebook Structure
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Amazon RecSys GNN - SimGCL Training\n",
                    "## Simple Graph Contrastive Learning with Weighted BPR Loss\n",
                    "\n",
                    "This notebook implements:\n",
                    "- **SimGCL**: Contrastive Learning to overcome data sparsity\n",
                    "- **Weighted BPR Loss**: Rating information as loss weights\n",
                    "- **Combined Strategy**: Sparsity solution + Rating utilization"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Imports & Setup\n",
                    "import numpy as np\n",
                    "import pandas as pd\n",
                    "import torch\n",
                    "import torch.nn as nn\n",
                    "import torch.nn.functional as F\n",
                    "from torch.optim import AdamW\n",
                    "from torch.optim.lr_scheduler import CosineAnnealingLR\n",
                    "import matplotlib.pyplot as plt\n",
                    "import pickle\n",
                    "import time\n",
                    "import warnings\n",
                    "import os\n",
                    "\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Create directories\n",
                    "os.makedirs('models', exist_ok=True)\n",
                    "os.makedirs('outputs', exist_ok=True)\n",
                    "\n",
                    "# Random Seed\n",
                    "SEED = 42\n",
                    "np.random.seed(SEED)\n",
                    "torch.manual_seed(SEED)\n",
                    "\n",
                    "# Device\n",
                    "if torch.cuda.is_available():\n",
                    "    device = torch.device('cuda')\n",
                    "else:\n",
                    "    device = torch.device('cpu')\n",
                    "print(f\"Device: {device}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Model Definitions (SimGCL)\n",
                    model_source
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Training Pipeline (SimGCL)\n",
                    train_source
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "---\n",
                    "## Inference (추론)\n",
                    "학습된 모델로 테스트 데이터에 대한 추천을 생성합니다."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Inference System\n",
                    "class SimGCLInference:\n",
                    "    def __init__(self, model_path, data_dir, graph_data, user2idx, item2idx, user_k, user_train_items):\n",
                    "        self.device = device\n",
                    "        \n",
                    "        # Load model checkpoint\n",
                    "        checkpoint = torch.load(model_path, map_location=self.device)\n",
                    "        config = checkpoint['config']\n",
                    "        \n",
                    "        # Store mappings\n",
                    "        self.user2idx = user2idx\n",
                    "        self.item2idx = item2idx\n",
                    "        self.user_k = user_k\n",
                    "        self.user_train_items = user_train_items\n",
                    "        \n",
                    "        # Load graph\n",
                    "        self.edge_index = graph_data['edge_index'].to(self.device)\n",
                    "        self.edge_weight = graph_data['cca_weight'].to(self.device)\n",
                    "        \n",
                    "        # Initialize model\n",
                    "        self.model = LightGCN_SimGCL(\n",
                    "            config['n_users'],\n",
                    "            config['n_items'],\n",
                    "            config['emb_dim'],\n",
                    "            config['n_layers'],\n",
                    "            eps=config['eps']\n",
                    "        ).to(self.device)\n",
                    "        \n",
                    "        self.model.load_state_dict(checkpoint['state_dict'])\n",
                    "        self.model.eval()\n",
                    "        \n",
                    "        print('✅ SimGCL Inference 시스템 초기화 완료')\n",
                    "        \n",
                    "        # Pre-compute embeddings\n",
                    "        with torch.no_grad():\n",
                    "            self.user_emb, self.item_emb = self.model(self.edge_index, self.edge_weight, perturbed=False)\n",
                    "    \n",
                    "    def predict(self, test_df):\n",
                    "        results = []\n",
                    "        \n",
                    "        for user_id, group in test_df.groupby('user'):\n",
                    "            # Unknown user\n",
                    "            if user_id not in self.user2idx:\n",
                    "                for _, row in group.iterrows():\n",
                    "                    results.append({'user': row['user'], 'item': row['item'], 'recommend': 'X'})\n",
                    "                continue\n",
                    "            \n",
                    "            u_idx = self.user2idx[user_id]\n",
                    "            K = self.user_k.get(u_idx, 2)\n",
                    "            MIN_K = 2\n",
                    "            \n",
                    "            items_to_score = []\n",
                    "            \n",
                    "            for _, row in group.iterrows():\n",
                    "                item_id = row['item']\n",
                    "                \n",
                    "                if item_id not in self.item2idx:\n",
                    "                    results.append({'user': row['user'], 'item': row['item'], 'recommend': 'X'})\n",
                    "                    continue\n",
                    "                \n",
                    "                i_idx = self.item2idx[item_id]\n",
                    "                \n",
                    "                # Skip already purchased items\n",
                    "                if i_idx in self.user_train_items.get(u_idx, set()):\n",
                    "                    results.append({'user': row['user'], 'item': row['item'], 'recommend': 'X'})\n",
                    "                    continue\n",
                    "                \n",
                    "                items_to_score.append((i_idx, row))\n",
                    "            \n",
                    "            if not items_to_score:\n",
                    "                continue\n",
                    "            \n",
                    "            # Batch scoring\n",
                    "            item_indices = torch.LongTensor([i for i, _ in items_to_score]).to(self.device)\n",
                    "            \n",
                    "            with torch.no_grad():\n",
                    "                scores = (self.user_emb[u_idx] * self.item_emb[item_indices]).sum(dim=1).cpu().numpy()\n",
                    "            \n",
                    "            # Top-K selection (README 50% rule)\n",
                    "            num_recommend = max(MIN_K, min(K, len(scores) // 2))\n",
                    "            top_indices = set(np.argsort(scores)[-num_recommend:])\n",
                    "            \n",
                    "            for idx, (item_idx, row) in enumerate(items_to_score):\n",
                    "                recommend = 'O' if idx in top_indices else 'X'\n",
                    "                results.append({'user': row['user'], 'item': row['item'], 'recommend': recommend})\n",
                    "        \n",
                    "        return pd.DataFrame(results)\n",
                    "\n",
                    "print('Inference class defined successfully!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run Inference on Test Set\n",
                    "print('\\n=== 추론 시작 ===')\n",
                    "\n",
                    "# Initialize inference system\n",
                    "inference = SimGCLInference(\n",
                    "    model_path='models/simgcl_best.pt',\n",
                    "    data_dir='/kaggle/input/amazon/',\n",
                    "    graph_data=graph_data,\n",
                    "    user2idx=user2idx,\n",
                    "    item2idx=item2idx,\n",
                    "    user_k=user_k,\n",
                    "    user_train_items=user_train_items\n",
                    ")\n",
                    "\n",
                    "# Load test data (use validation split as example)\n",
                    "test_df = val_df[['user', 'item']].copy()\n",
                    "\n",
                    "# Generate predictions\n",
                    "predictions = inference.predict(test_df)\n",
                    "\n",
                    "# Display sample results\n",
                    "print('\\n샘플 추천 결과:')\n",
                    "print('='*50)\n",
                    "print(f\"{'user':<15}{'item':<15}{'recommend':<10}\")\n",
                    "for _, row in predictions.head(20).iterrows():\n",
                    "    print(f\"{str(row['user']):<15}{str(row['item']):<15}{row['recommend']:<10}\")\n",
                    "print('='*50)\n",
                    "\n",
                    "total_cnt = len(predictions)\n",
                    "rec_cnt = len(predictions[predictions['recommend'] == 'O'])\n",
                    "print(f'\\nTotal recommends: {rec_cnt}/{total_cnt} ({rec_cnt/total_cnt*100:.1f}%)')\n",
                    "print(f'Not recommend: {total_cnt - rec_cnt}/{total_cnt}')\n",
                    "\n",
                    "# Save results\n",
                    "predictions.to_csv('outputs/predictions.csv', index=False)\n",
                    "print('\\n✅ 추론 완료! 결과 저장: outputs/predictions.csv')"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # 5. Write Notebook
    output_path = 'kaggle_train_simgcl.ipynb'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"✅ SimGCL Kaggle notebook generated: {output_path}")

if __name__ == "__main__":
    create_simgcl_notebook()
