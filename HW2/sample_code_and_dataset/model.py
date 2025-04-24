from matplotlib.pylab import f
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv ,GATConv
from dgl.nn import SAGEConv,GINConv

class GCN(nn.Module):
    """
    Baseline Model:
    - A simple two-layer GCN model, similar to https://github.com/tkipf/pygcn
    - Implement with DGL package
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(
            GraphConv(in_size, hid_size, activation=F.relu)
        )
        self.layers.append(GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

class StableGATModel(nn.Module):
    """
    Two-layer GAT + One Linear Layer for stable performance.
    GAT for rich structural learning; Linear for simpler final classification.
    """
    def __init__(self, in_size, hid_size, out_size, num_heads=2):
        super().__init__()
        self.gat1 = GATConv(in_size, hid_size, num_heads=num_heads)
        self.gat2 = GATConv(hid_size * num_heads, hid_size, num_heads=num_heads)
        self.linear = nn.Linear(hid_size * num_heads, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = self.gat1(g, features)      # (N, num_heads, hid_size)
        h = h.flatten(1)                # → (N, hid_size * num_heads)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.gat2(g, h)             # → (N, num_heads, hid_size)
        h = h.flatten(1)                # → (N, hid_size * num_heads again)
        h = F.elu(h)
        h = self.dropout(h)

        out = self.linear(h)            # → (N, out_size)
        return out    
class GATModel(nn.Module):
    """
    Enhanced GAT Model:
    - 3-layer GAT with ELU, Dropout, and optional BatchNorm
    - Hidden dim and head number can be configured
    """
    def __init__(self, in_size, hid_size, out_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(0.5)

        # Layer 1: in_size → hid_size * num_heads
        self.gat1 = GATConv(in_size, hid_size, num_heads=num_heads)
        # self.bn1 = nn.BatchNorm1d(hid_size * num_heads)  # Optional

        # Layer 2: hid_size * num_heads → hid_size * num_heads
        self.gat2 = GATConv(hid_size * num_heads, hid_size, num_heads=num_heads)
        # self.bn2 = nn.BatchNorm1d(hid_size * num_heads)

        # Layer 3: hid_size * num_heads → out_size (final classification)
        self.gat3 = GATConv(hid_size * num_heads, out_size, num_heads=1)

    def forward(self, g, features):
        h = self.gat1(g, features)      # (N, num_heads, hid_size)
        h = h.flatten(1)                # → (N, hid_size * num_heads)
        h = F.elu(h)
        # h = self.bn1(h)
        h = self.dropout(h)

        h = self.gat2(g, h)
        h = h.flatten(1)
        h = F.elu(h)
        # h = self.bn2(h)
        h = self.dropout(h)

        h = self.gat3(g, h)             # (N, 1, out_size)
        return h.squeeze(1)             # → (N, out_size)

class YourGNNModel(nn.Module):

    """
    A 3-layer GraphSAGE model with mean aggregator.
    Designed to outperform the simple GCN baseline.
    """
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.conv1 = SAGEConv(in_size, hid_size, aggregator_type='mean')
        self.conv2 = SAGEConv(hid_size, hid_size, aggregator_type='mean')
        self.conv3 = SAGEConv(hid_size, out_size, aggregator_type='mean')
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv2(g, h)
        h = F.relu(h)
        h = self.dropout(h)

        h = self.conv3(g, h)
        return h

class SSP_GCN(nn.Module):
    def __init__(self, encoder, hid_size, out_size):
        super().__init__()
        self.encoder = encoder
        self.classify = GraphConv(hid_size, out_size)
    def forward(self, g, x):
        h = self.encoder(g, x)          # 使用預訓練好的 encoder
        return self.classify(g, h)      # 新增分類層   
    
class GCNEncoder(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.conv1 = GraphConv(in_size, hid_size, activation=F.relu)
        self.conv2 = GraphConv(hid_size, hid_size)
    def forward(self, g, x):
        h = F.relu(self.conv1(g, x))
        return self.conv2(g, h)

# Contrastive head
class ContrastiveHead(nn.Module):
    def __init__(self, hid_size, proj_size):
        super().__init__()
        self.fc1 = nn.Linear(hid_size, proj_size)
        self.fc2 = nn.Linear(proj_size, proj_size)
    def forward(self, h):
        z = F.relu(self.fc1(h))
        return self.fc2(z)
    
 # SSP   
class SAGEEncoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        # 第一層
        self.convs.append(SAGEConv(in_size, hid_size, aggregator_type='mean'))
        # 中間層
        for _ in range(num_layers-2):
            self.convs.append(SAGEConv(hid_size, hid_size, aggregator_type='mean'))
        # 最後一層
        self.convs.append(SAGEConv(hid_size, out_size, aggregator_type='mean'))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, feat):
        h = feat
        for i, conv in enumerate(self.convs):
            h = conv(g, h)
            if i != len(self.convs)-1:
                h = F.relu(h)
                h = self.dropout(h)
        return h                       # 節點嵌入

class SSPModel(nn.Module):
    """
    encoder + projector（預訓練用）
    fine-tune 階段把 projector 捨棄，另接 LinearCls
    """
    def __init__(self, in_size, hid_size, proj_size=128):
        super().__init__()
        self.encoder = SAGEEncoder(in_size, hid_size, hid_size)
        self.projector = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.PReLU(),
            nn.Linear(hid_size, proj_size)
        )

    def forward(self, g, x):
        z = self.encoder(g, x)         # [N, hid]
        p = self.projector(z)          # [N, proj]
        p = F.normalize(p, dim=1)
        return z, p