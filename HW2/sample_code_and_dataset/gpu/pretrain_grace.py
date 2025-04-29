# pretrain_grace.py
import torch
import torch.nn.functional as F
from torch import nn
import dgl
import random
from data_loader import load_data
from grace_model import GCNEncoder, GRACE

def drop_edge(g, drop_prob=0.1):
    num_edges = g.num_edges()
    mask = torch.rand(num_edges, device=g.device) > drop_prob
    eids = torch.arange(num_edges, device=g.device)[mask]
    new_g = dgl.edge_subgraph(g, eids, relabel_nodes=False)  # ✅ 保留所有節點
    return dgl.add_self_loop(new_g)

def feature_dropout(x, drop_prob=0.1):
    drop_mask = torch.rand(x.shape, device=x.device) > drop_prob
    return x * drop_mask.float()

def nt_xent(p1, p2, temperature=0.2):
    z1 = F.normalize(p1, dim=1)
    z2 = F.normalize(p2, dim=1)
    N = z1.shape[0]

    sim_matrix = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(N, device=z1.device)
    loss1 = F.cross_entropy(sim_matrix, labels)
    loss2 = F.cross_entropy(sim_matrix.T, labels)
    return (loss1 + loss2) / 2

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    feat, g, num_classes, *_ = load_data()
    feat = feat.to(device)
    g = g.to(device)

    # encoder = GCNEncoder(feat.shape[1], 256, 256).to(device)
    # model = GRACE(encoder, 256).to(device)
    encoder = GCNEncoder(feat.shape[1], 512, 512).to(device)
    model = GRACE(encoder, 512).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 301):
        model.train()
        g1 = drop_edge(g, 0.1)
        g2 = drop_edge(g, 0.1)
        x1 = feature_dropout(feat, 0.1)
        x2 = feature_dropout(feat, 0.1)
        p1, p2 = model(g1, x1, g2, x2)
        loss = nt_xent(p1, p2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"[Pretrain] Epoch {epoch:03d} | Contrastive Loss: {loss.item():.4f}")

    torch.save(encoder.state_dict(), 'encoder_ssp.pth')
