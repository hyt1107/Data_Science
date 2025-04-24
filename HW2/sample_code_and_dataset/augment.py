import torch
import dgl
import random

def drop_edge_oldver(g, drop_prob=0.2):
    num_edges = g.number_of_edges()
    mask = torch.rand(num_edges, device=g.device) >= drop_prob
    eids = torch.nonzero(mask, as_tuple=False).squeeze()
    return dgl.edge_subgraph(g, eids, relabel_nodes=True)

def mask_feature(x, mask_prob=0.05):
    mask = torch.rand_like(x) >= mask_prob
    return x * mask


def drop_edge(g, drop_prob=0.05):
    num_edges = g.num_edges()
    # TRUE 表示「要被刪掉」
    mask = torch.rand(num_edges, device=g.device) < drop_prob
    drop_eids = torch.nonzero(mask, as_tuple=False).squeeze()

    # 刪掉邊後回傳一張『仍含 20000 個節點』的新圖
    return dgl.remove_edges(g, drop_eids)
