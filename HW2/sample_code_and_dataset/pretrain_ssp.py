import torch, torch.nn.functional as F, argparse, os
from data_loader import load_data
from model import SSPModel
from augment import drop_edge, mask_feature

def info_nce(p1, p2, temperature=0.7):
    sim = torch.mm(p1, p2.t()) / temperature        # [N,N]
    labels = torch.arange(p1.size(0), device=p1.device)
    # ==== 對稱（雙向）Cross-Entropy ====
    loss_fwd = F.cross_entropy(sim, labels)      # p1 → p2
    loss_bwd = F.cross_entropy(sim.t(), labels)  # p2 → p1
    loss = 0.5 * (loss_fwd + loss_bwd)
    return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save', type=str, default='encoder_ssp.pth')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    dev = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')
    feat, g, _, _, _, _, _, _, _ = load_data()
    feat, g = feat.to(dev), g.to(dev)

    model = SSPModel(feat.shape[1], hid_size=64).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        model.train()
        # 產生兩個增強視角
        g1, g2 = drop_edge(g), drop_edge(g)
        x1, x2 = mask_feature(feat), mask_feature(feat)

        _, p1 = model(g1, x1)
        _, p2 = model(g2, x2)
        loss = info_nce(p1, p2)
        

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        opt.step()
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1:03d}  SSL loss={loss.item():.4f}')

    torch.save(model.encoder.state_dict(), args.save)
    print('✓ Pre-training finished, encoder saved.')
