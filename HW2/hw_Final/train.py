import random
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from data_loader import load_data
from model import GCNEncoder, GRACE, GCN_LPA

def safe_one_hot(labels, num_classes):
    valid = labels >= 0
    labels_clone = labels.clone()
    labels_clone[~valid] = 0  # 把無效的 label 暫時填成 0
    one_hot = F.one_hot(labels_clone, num_classes=num_classes).float()
    one_hot[~valid] = 0  # 把原本無標籤的位置 one-hot 全部設成 0
    return one_hot

def compute_loss(pred_logits, lpa_preds, labels, mask, lpa_weight):
    # GCN 的交叉熵損失
    loss_gcn = F.cross_entropy(pred_logits[mask], labels[mask])

    # LPA 的損失（例如，均方誤差）
    loss_lpa = F.mse_loss(lpa_preds[mask], F.one_hot(labels[mask], num_classes=pred_logits.size(1)).float())

    # 總損失
    total_loss = loss_gcn + lpa_weight * loss_lpa
    return total_loss

def drop_edge(g: dgl.DGLGraph, low: float, high: float) -> dgl.DGLGraph:
    """Randomly drop edges in [low, high] range and add self-loops."""
    p = random.uniform(low, high)
    mask = torch.rand(g.num_edges(), device=g.device) > p
    eids = torch.arange(g.num_edges(), device=g.device)[mask]
    sub = dgl.edge_subgraph(g, eids, relabel_nodes=False)
    return dgl.add_self_loop(sub)

def feature_dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    """Element-wise dropout on features."""
    mask = torch.rand_like(x) > p
    return x * mask.float()

def nt_xent(p1: torch.Tensor, p2: torch.Tensor, temp: float) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss."""
    z1 = F.normalize(p1, dim=1)
    z2 = F.normalize(p2, dim=1)
    sim = torch.mm(z1, z2.T) / temp
    labels = torch.arange(z1.size(0), device=z1.device)
    loss1 = F.cross_entropy(sim, labels)
    loss2 = F.cross_entropy(sim.T, labels)
    return 0.5 * (loss1 + loss2)

def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds[mask] == labels[mask]).float().mean().item()

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        self._load_data()
        self._build_model()

    def _load_data(self):
        feat, g, num_cls, tr_lbl, vl_lbl, te_lbl, tr_m, vl_m, te_m = load_data()
        self.feat = feat.to(self.device)
        self.g = g.to(self.device)
        self.num_classes = num_cls
        self.train_mask = torch.tensor(tr_m, dtype=torch.bool, device=self.device)
        self.val_mask = torch.tensor(vl_m, dtype=torch.bool, device=self.device)
        self.test_mask = torch.tensor(te_m, dtype=torch.bool, device=self.device)

        # 初始化 labels
        # labels = torch.zeros((self.feat.size(0),), dtype=torch.long, device=self.device)
        labels = torch.full((self.feat.size(0),), -1, dtype=torch.long, device=self.device)

        # 如果是 numpy，轉成 tensor；如果是 tensor，直接用
        if not torch.is_tensor(tr_lbl):
            tr_lbl = torch.from_numpy(tr_lbl)
        if not torch.is_tensor(vl_lbl):
            vl_lbl = torch.from_numpy(vl_lbl)
        if not torch.is_tensor(te_lbl):
            te_lbl = torch.from_numpy(te_lbl)

        labels[self.train_mask] = tr_lbl.long().to(self.device)
        labels[self.val_mask] = vl_lbl.long().to(self.device)
        labels[self.test_mask] = te_lbl.long().to(self.device)

        self.labels = labels



    def _build_model(self):
        if self.args.model == "grace":
            hid = self.args.hidden_dim
            encoder = GCNEncoder(self.feat.size(1), hid, hid, dropout=self.args.dropout).to(self.device)
            self.model = GRACE(encoder, hid).to(self.device)
        elif self.args.model == "gcn_lpa":
            self.model = GCN_LPA(
                in_feats=self.feat.size(1),
                hidden_feats=self.args.hidden_dim,
                out_feats=self.num_classes,
                dropout=self.args.dropout,
                lpa_iters=self.args.lpa_iters,
                lpa_weight=self.args.lpa_weight
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported model type: {self.args.model}")

    def pretrain(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_pretrain, weight_decay=self.args.wd_pretrain)
        self.model.train()
        for epoch in range(1, self.args.pretrain_epochs + 1):
            g1 = drop_edge(self.g, self.args.edge_drop, self.args.max_edge_drop)
            g2 = drop_edge(self.g, self.args.edge_drop, self.args.max_edge_drop)
            x1 = feature_dropout(self.feat, self.args.feat_drop)
            x2 = feature_dropout(self.feat, self.args.feat_drop)
            p1, p2 = self.model(g1, x1, g2, x2)
            loss = nt_xent(p1, p2, self.args.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch == 1 or epoch % 20 == 0:
                print(f"[Pretrain] Epoch {epoch:03d} | Loss {loss.item():.4f}")
        torch.save(self.model.encoder.state_dict(), "encoder_grace.pth")

    def finetune(self):
        if self.args.model == "grace":
            encoder = self.model.encoder
            encoder.eval()
            for p in encoder.parameters():
                p.requires_grad = False
            hid = self.args.hidden_dim
            # classifier = nn.Sequential(
            #     nn.Linear(hid, hid // 2),
            #     nn.ReLU(),
            #     nn.Dropout(self.args.cls_dropout),
            #     nn.Linear(hid // 2, self.num_classes)
            # ).to(self.device)

            classifier = nn.Sequential(
                nn.Linear(hid, hid // 2),
                nn.ReLU(),
                nn.Dropout(self.args.cls_dropout),
                nn.Linear(hid // 2, self.num_classes)
            ).to(self.device)


            opt = torch.optim.Adam(classifier.parameters(), lr=self.args.lr_finetune, weight_decay=self.args.wd_finetune)
            loss_fn = nn.CrossEntropyLoss()
            best_acc, patience = 0.0, 0
            for epoch in range(1, self.args.finetune_epochs + 1):
                classifier.train()
                with torch.no_grad():
                    h = encoder(self.g, self.feat)
                logits = classifier(h)
                loss = loss_fn(logits[self.train_mask], self.labels[self.train_mask])
                opt.zero_grad()
                loss.backward()
                opt.step()
                val_acc = accuracy(logits, self.labels, self.val_mask)
                if epoch == 1 or epoch % 10 == 0:
                    print(f"[Finetune] Epoch {epoch:03d} | Loss {loss.item():.4f} | ValAcc {val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience = 0
                    torch.save(classifier.state_dict(), "best_classifier.pth")
                    print(f"Best acc:{best_acc:.4f}!!")
                else:
                    patience += 1
                    if patience >= self.args.patience:
                        print(f"Early stop at epoch {epoch}")
                        break
            classifier.load_state_dict(torch.load("best_classifier.pth"))
            self.classifier = classifier
        elif self.args.model == "gcn_lpa":
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_finetune, weight_decay=self.args.wd_finetune)
            best_acc, patience = 0.0, 0
            for epoch in range(1, self.args.finetune_epochs + 1):
                optimizer.zero_grad()
                logits, lpa_preds = self.model(self.g, self.feat, safe_one_hot(self.labels, num_classes=self.num_classes), self.train_mask)
                loss = compute_loss(logits, lpa_preds, self.labels, self.train_mask, self.args.lpa_weight)
                loss.backward()
                optimizer.step()
                val_acc = accuracy(logits, self.labels, self.val_mask)
                if epoch == 1 or epoch % 10 == 0:
                    print(f"[GCN-LPA] Epoch {epoch:03d} | Loss {loss.item():.4f} | ValAcc {val_acc:.4f}")
                if val_acc > best_acc:
                    best_acc = val_acc
                    patience = 0
                    torch.save(self.model.state_dict(), "best_gcn_lpa.pth")
                    print(f"Best acc:{best_acc:.4f}!!")
                else:
                    patience += 1
                    if patience >= self.args.patience:
                        print(f"Early stop at epoch {epoch}")
                        break
            self.model.load_state_dict(torch.load("best_gcn_lpa.pth"))
        else:
            raise ValueError(f"Unsupported model type: {self.args.model}")

    def export_csv(self, path: str | Path = "output.csv"):
        self.model.eval()
        with torch.no_grad():
            if self.args.model == "grace":
                h = self.model.encoder(self.g, self.feat)
                preds = self.classifier(h)[self.test_mask].argmax(dim=1)
            elif self.args.model == "gcn_lpa":
                logits, _ = self.model(self.g, self.feat, safe_one_hot(self.labels, num_classes=self.num_classes), self.train_mask)
                preds = logits[self.test_mask].argmax(dim=1)
            else:
                raise ValueError(f"Unsupported model type: {self.args.model}")
        with open(path, "w") as f:
            f.write("ID,Predict\n")
            for i, p in enumerate(preds):
                f.write(f"{i},{int(p)}\n")
        print(f"✅ {path} generated!")

    def run_full(self):
        if self.args.model == "grace":
            print("=== Self-supervised Pretraining ===")
            self.pretrain()
            print("\n=== Supervised Finetuning ===")
            self.finetune()
            self.export_csv()
        elif self.args.model == "gcn_lpa":
            print("=== Training GCN-LPA ===")
            self.finetune()
            self.export_csv(path="output_gcn_lpa.csv")
        else:
            raise ValueError(f"Unsupported model type: {self.args.model}")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Model selection
    parser.add_argument("--model", type=str, default="grace", choices=["grace", "gcn_lpa"],
                        help="Model to use: grace or gcn_lpa")

    # Pretrain args (only for GRACE)
    parser.add_argument("--pretrain_epochs", type=int, default=350)
    parser.add_argument("--edge_drop", type=float, default=0.05)
    parser.add_argument("--max_edge_drop", type=float, default=0.15)
    parser.add_argument("--feat_drop", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--lr_pretrain", type=float, default=1e-3)
    parser.add_argument("--wd_pretrain", type=float, default=0.0)

    # Finetune args (common for both models)
    parser.add_argument("--finetune_epochs", type=int, default=200)
    parser.add_argument("--lr_finetune", type=float, default=2e-3)
    parser.add_argument("--wd_finetune", type=float, default=1e-3)
    parser.add_argument("--cls_dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=30)

    # GCN-LPA specific args
    parser.add_argument("--lpa_iters", type=int, default=10, help="Number of LPA iterations (only for GCN-LPA)")
    parser.add_argument("--lpa_weight", type=float, default=0.5, help="LPA loss weight (only for GCN-LPA)")

    # Miscellaneous args
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_gpu", action="store_true")
    
    args = parser.parse_args()
    trainer = Trainer(args)
    trainer.run_full()