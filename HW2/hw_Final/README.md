# README.md

113164522 黃鈺婷

# 執行方法

## 使用GRACE (Self-supervised Pretrain + Finetune)

```bash
python train.py --model grace --use_gpu --edge_drop 0.1 --max_edge_drop 0.1 --dropout 0.4
```

- 預訓練：對比式學習 (Contrastive Learning)
- 微調：分類器訓練
- 輸出檔案：`output.csv`



# 環境

| 項目 | 版本 / 說明 |
| --- | --- |
| Python | 3.10 |
| NumPy | 1.24.3 |
| scikit-learn | 最新版（自動下載，目前對應 1.3.x） |
| PyTorch | 2.2.0 （支援 CUDA 12.1） |
| TorchVision | 0.17.0 |
| TorchAudio | 2.2.0 |
| DGL (Deep Graph Library) | 2.0.0 （支援 CUDA 12.1） |
| SciPy | 最新版（通常 1.10.x 或以上） |
| NetworkX | 最新版（通常 3.x） |
| setuptools、packaging | 用於支援安裝 |

```bash
conda create -n gnn_env python=3.10
conda activate gnn_env
pip install numpy==1.24.3
pip install scikit-learn
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install setuptools packaging
```

```bash
pip install scipy networkx
pip install dgl-2.0.0+cu121-cp310-cp310-win_amd64.whl
```