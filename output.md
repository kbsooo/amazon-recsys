============================================================
5. CCA 모델 학습 (Binary Recommendation)
============================================================
학습 시작...
Epoch 5/100 | Loss: 0.0969 | Val Recall@20: 0.3965 | Val NDCG@20: 0.1779
Epoch 10/100 | Loss: 0.0461 | Val Recall@20: 0.4311 | Val NDCG@20: 0.1966
Epoch 15/100 | Loss: 0.0279 | Val Recall@20: 0.4515 | Val NDCG@20: 0.2089
Epoch 20/100 | Loss: 0.0185 | Val Recall@20: 0.4651 | Val NDCG@20: 0.2166
Epoch 25/100 | Loss: 0.0127 | Val Recall@20: 0.4745 | Val NDCG@20: 0.2235
Epoch 30/100 | Loss: 0.0091 | Val Recall@20: 0.4806 | Val NDCG@20: 0.2276
Epoch 35/100 | Loss: 0.0067 | Val Recall@20: 0.4860 | Val NDCG@20: 0.2312
Epoch 40/100 | Loss: 0.0053 | Val Recall@20: 0.4894 | Val NDCG@20: 0.2336
Epoch 45/100 | Loss: 0.0044 | Val Recall@20: 0.4917 | Val NDCG@20: 0.2359
Epoch 50/100 | Loss: 0.0036 | Val Recall@20: 0.4934 | Val NDCG@20: 0.2370
Epoch 55/100 | Loss: 0.0032 | Val Recall@20: 0.4952 | Val NDCG@20: 0.2382
Epoch 60/100 | Loss: 0.0029 | Val Recall@20: 0.4962 | Val NDCG@20: 0.2384
Epoch 65/100 | Loss: 0.0028 | Val Recall@20: 0.4972 | Val NDCG@20: 0.2389
Epoch 70/100 | Loss: 0.0027 | Val Recall@20: 0.4977 | Val NDCG@20: 0.2392
Epoch 75/100 | Loss: 0.0025 | Val Recall@20: 0.4980 | Val NDCG@20: 0.2395
Epoch 80/100 | Loss: 0.0024 | Val Recall@20: 0.4980 | Val NDCG@20: 0.2396
Epoch 85/100 | Loss: 0.0023 | Val Recall@20: 0.4982 | Val NDCG@20: 0.2397
Epoch 90/100 | Loss: 0.0024 | Val Recall@20: 0.4983 | Val NDCG@20: 0.2398
Epoch 95/100 | Loss: 0.0024 | Val Recall@20: 0.4983 | Val NDCG@20: 0.2398
Epoch 100/100 | Loss: 0.0024 | Val Recall@20: 0.4983 | Val NDCG@20: 0.2398

============================================================
6. CCB 모델 학습 (Rating Prediction + BPR)
============================================================
학습 시작...
Epoch 5/100 | Total Loss: 0.4575 | BPR: 0.2659 | MSE: 0.3832 | Val RMSE: 2.2258
Epoch 10/100 | Total Loss: 0.2578 | BPR: 0.1817 | MSE: 0.1522 | Val RMSE: 2.3175
Epoch 15/100 | Total Loss: 0.1872 | BPR: 0.1322 | MSE: 0.1101 | Val RMSE: 2.2924
Epoch 20/100 | Total Loss: 0.1450 | BPR: 0.0998 | MSE: 0.0903 | Val RMSE: 2.3044
Epoch 25/100 | Total Loss: 0.1161 | BPR: 0.0774 | MSE: 0.0774 | Val RMSE: 2.3063
Epoch 30/100 | Total Loss: 0.0960 | BPR: 0.0616 | MSE: 0.0688 | Val RMSE: 2.3127
Epoch 35/100 | Total Loss: 0.0813 | BPR: 0.0499 | MSE: 0.0628 | Val RMSE: 2.3241
Epoch 40/100 | Total Loss: 0.0700 | BPR: 0.0410 | MSE: 0.0580 | Val RMSE: 2.3392
Epoch 45/100 | Total Loss: 0.0617 | BPR: 0.0343 | MSE: 0.0548 | Val RMSE: 2.3613
Epoch 50/100 | Total Loss: 0.0555 | BPR: 0.0293 | MSE: 0.0524 | Val RMSE: 2.3918
Epoch 55/100 | Total Loss: 0.0509 | BPR: 0.0255 | MSE: 0.0509 | Val RMSE: 2.4128
✅ CCB 모델 학습 완료 (Best Val RMSE: 2.2258)