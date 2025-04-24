# pretrain.py
import torch
import torch.nn.functional as F
from torch.optim import Adam
from dgl.sampling import sample_neighbors
from data_loader import load_data
from model import GCNEncoder, ContrastiveHead

# N×N的矩陣
def info_nce(z1, z2, temp=0.5):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    sim = torch.mm(z1, z2.t()) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
    return loss

def memory_efficient_info_nce(z1, z2, temp=0.5, chunk_size=2048):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    n = z1.size(0)
    
    # 分塊計算相似度矩陣
    sim_matrix = torch.zeros(n, n, device=z1.device)
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        chunk = z1[i:end]
        sim_matrix[i:end] = torch.mm(chunk, z2.t()) / temp
    
    labels = torch.arange(n, device=z1.device)
    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.t(), labels)) / 2
    return loss

def drop_feature(x, drop_ratio):
    mask = torch.rand_like(x) < drop_ratio
    x2 = x.clone()
    x2[mask] = 0
    return x2

if __name__ == '__main__':
    # 1. 載入資料
    features, graph, *_ = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, graph = features.to(device), graph.to(device)


    # 2. 建模 & 優化器
    in_size, hid_size, proj_size = features.shape[1], 64, 32  #128 ,64
    encoder = GCNEncoder(in_size, hid_size).to(device)
    head    = ContrastiveHead(hid_size, proj_size).to(device)
    opt     = Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-3)

    # 3. SSL 訓練
    pre_epochs, mask_ratio = 200, 0.3
    for epoch in range(pre_epochs):
        encoder.train(); head.train()
        # 兩種隨機遮擋
        xa = drop_feature(features, mask_ratio)
        xb = drop_feature(features, mask_ratio)
        ha = encoder(graph, xa)
        hb = encoder(graph, xb)
        za = head(ha); zb = head(hb)
        loss = memory_efficient_info_nce(za, zb)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 20 == 0:
            print(f"[Pretrain] Epoch {epoch} | SSL Loss {loss.item():.4f}")

    # 4. 儲存 encoder
    torch.save(encoder.state_dict(), "encoder_ssp.pth")
    print("Encoder pretraining done.")
