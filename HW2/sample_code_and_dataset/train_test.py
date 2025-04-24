import torch
import torch.nn as nn
from torch.optim import Adam
from argparse import ArgumentParser

from data_loader import load_data
from model import GCNEncoder, SSP_GCN


def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy on a given mask"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, preds = torch.max(logits, dim=1)
        correct = torch.sum(preds == labels).item()
        return correct / len(labels)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='encoder_ssp.pth',
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of fine-tune epochs')
    parser.add_argument('--es_iters', type=int, default=None,
                        help='Early stopping patience (None to disable)')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU if available')
    args = parser.parse_args()

    # 設備
    device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    # 載入資料
    feat, graph, num_classes, train_labels, val_labels, test_labels, \
        train_mask, val_mask, test_mask = load_data()
    feat, graph = feat.to(device), graph.to(device)

    # 1) 載入並構建 encoder
    hid_size = 64  # 一定要和 pretrain.py 保持一致
    encoder = GCNEncoder(in_size=feat.shape[1], hid_size=hid_size).to(device)
    encoder.load_state_dict(torch.load(args.pretrained, map_location=device))

    # 2) 組成 SSP_GCN
    model = SSP_GCN(encoder=encoder, hid_size=hid_size, out_size=num_classes).to(device)

    # 3) 優化器 & loss
    optimizer = Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    loss_fcn = nn.CrossEntropyLoss()

    # Early stopping 參數
    best_loss, es_count = float('inf'), 0

    # 4) 微調迴圈
    for epoch in range(args.epochs):
        model.train()
        logits = model(graph, feat)
        loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 驗證
        val_logits = logits[val_mask]
        val_loss = loss_fcn(val_logits, val_labels).item()
        val_acc = evaluate(graph, feat, val_labels, val_mask, model)
        print(f"Epoch {epoch:03d} | Train Loss {loss:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        # Early stopping
        if args.es_iters is not None:
            if val_loss < best_loss:
                best_loss, es_count = val_loss, 0
            else:
                es_count += 1
            if es_count >= args.es_iters:
                print(f"Early stopping at epoch {epoch}")
                break

    # 5) 測試 & 匯出 CSV
    model.eval()
    with torch.no_grad():
        test_logits = model(graph, feat)[test_mask]
        preds = test_logits.argmax(dim=1).cpu().numpy()

    with open('output.csv', 'w') as f:
        f.write('ID,Predict\n')
        for i, p in enumerate(preds):
            f.write(f"{i},{int(p)}\n")

    print("Saved output.csv. Ready for Kaggle submission.")
