"""
최종 통합 노트북 생성 스크립트

전체 파이프라인을 포함한 Jupyter Notebook 생성
"""

import json

# 노트북 셀 정의
cells = []

# ==== TITLE ====
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Amazon GNN 추천 시스템 - 최종 통합 파이프라인\\n",
        "\\n",
        "## 프로젝트 개요\\n",
        "\\n",
        "**목표**: Amazon 사용자-아이템 구매 데이터를 활용한 GNN 기반 추천 시스템 구축\\n",
        "\\n",
        "**핵심 특징**:\\n",
        "- 사용자별 맞춤 추천 개수 (기존 interaction의 50% 이하)\\n",
        "- Cold-start 사용자 (≤10개) 무조건 2개 추천\\n",
        "- 3가지 GNN 모델 비교: LightGCN, NGCF, SimpleGCN\\n",
        "- 다양한 평가 지표 및 시각화\\n",
        "\\n",
        "**데이터셋**:\\n",
        "- 568,263 interactions\\n",
        "- 256,009 users\\n",
        "- 74,233 items\\n",
        "- Sparsity: 99.997%"
    ]
})

# ==== TOC ====
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 목차\\n",
        "\\n",
        "1. [환경 설정](#1.-환경-설정)\\n",
        "2. [데이터 로딩 및 전처리](#2.-데이터-로딩-및-전처리)\\n",
        "3. [데이터 분석 (EDA)](#3.-데이터-분석-(EDA))\\n",
        "4. [그래프 구축](#4.-그래프-구축)\\n",
        "5. [모델 구현](#5.-모델-구현)\\n",
        "6. [모델 학습](#6.-모델-학습)\\n",
        "7. [모델 평가](#7.-모델-평가)\\n",
        "8. [추천 생성](#8.-추천-생성)\\n",
        "9. [결과 분석 및 시각화](#9.-결과-분석-및-시각화)\\n",
        "10. [결론](#10.-결론)"
    ]
})

# ==== 1. 환경 설정 ====
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. 환경 설정"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import numpy as np\\n",
        "import pandas as pd\\n",
        "import matplotlib.pyplot as plt\\n",
        "import seaborn as sns\\n",
        "import torch\\n",
        "import torch.nn as nn\\n",
        "from tqdm import tqdm\\n",
        "import warnings\\n",
        "import time\\n",
        "\\n",
        "warnings.filterwarnings('ignore')\\n",
        "\\n",
        "# Matplotlib 설정\\n",
        "plt.rcParams['figure.figsize'] = (12, 6)\\n",
        "plt.rcParams['font.family'] = 'AppleGothic'\\n",
        "plt.rcParams['axes.unicode_minus'] = False\\n",
        "sns.set_style('whitegrid')\\n",
        "\\n",
        "# Device 설정\\n",
        "if torch.cuda.is_available():\\n",
        "    device = 'cuda'\\n",
        "elif torch.backends.mps.is_available():\\n",
        "    device = 'mps'\\n",
        "else:\\n",
        "    device = 'cpu'\\n",
        "\\n",
        "print(f'사용 디바이스: {device}')\\n",
        "print(f'PyTorch 버전: {torch.__version__}')\\n",
        "print('환경 설정 완료!')"
    ]
})

# 모듈 import
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 프로젝트 모듈 import\\n",
        "from preprocessing import DataPreprocessor, load_and_preprocess_data, create_edge_set\\n",
        "from models import LightGCN, NGCF, SimpleGCN\\n",
        "from train import GNNTrainer, BPRLoss\\n",
        "from evaluate import GNNEvaluator\\n",
        "from recommend import RecommendationSystem\\n",
        "\\n",
        "print('모든 모듈 로드 완료!')"
    ]
})

# ==== 2. 데이터 로딩 및 전처리 ====
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. 데이터 로딩 및 전처리"]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 데이터 로딩 및 전처리\\n",
        "preprocessor, train_df, val_df, test_df, k_values = load_and_preprocess_data(\\n",
        "    data_path='../data/amazon_train.csv',\\n",
        "    good_rating_threshold=4.0,\\n",
        "    cold_start_threshold=10,\\n",
        "    recommend_ratio=0.5,  # 50% 제한\\n",
        "    min_recommend=2,\\n",
        "    random_state=42\\n",
        ")\\n",
        "\\n",
        "n_users = preprocessor.n_users\\n",
        "n_items = preprocessor.n_items\\n",
        "\\n",
        "print(f'\\\\n사용자 수: {n_users:,}')\\n",
        "print(f'아이템 수: {n_items:,}')\\n",
        "print(f'학습 데이터: {len(train_df):,}')\\n",
        "print(f'검증 데이터: {len(val_df):,}')\\n",
        "print(f'테스트 데이터: {len(test_df):,}')"
    ]
})

# 계속 작성...
# 최종 노트북은 매우 길어질 것이므로 핵심 부분만 작성하고 
# 사용자가 필요시 확장할 수 있도록 템플릿 제공

print("최종 통합 노트북 생성 스크립트 준비 중...")
print("노트북이 매우 크므로 단계별로 생성합니다.")

# 노트북 저장
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_path = "amazon_recsys_final_template.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"템플릿 노트북 생성: {output_path}")
print("전체 노트북은 별도로 작성하겠습니다.")
