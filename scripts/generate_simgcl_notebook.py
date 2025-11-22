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
