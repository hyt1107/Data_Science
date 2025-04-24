import os
import warnings
from argparse import ArgumentParser

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from model import YourGNNModel, GATModel, StableGATModel
NWARNINGS = False
if NWARNINGS:
    warnings.filterwarnings("ignore")

def evaluate(g, features, labels, mask, model):
    """Evaluate model accuracy"""
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / labels.shape[0]

def train(g, features, train_labels, val_labels, train_mask, val_mask, model, epochs, es_iters=None):
    """
    Training loop with optional early stopping.
    """
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    if es_iters:
        print("Early stopping monitoring on")
        best_loss = float('inf')
        patience = 0

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_acc = evaluate(g, features, val_labels, val_mask, model)
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | Val Acc {val_acc:.4f}")

        if es_iters:
            val_loss = loss_fcn(logits[val_mask], val_labels).item()
            if val_loss < best_loss:
                best_loss = val_loss
                patience = 0
            else:
                patience += 1
            if patience >= es_iters:
                print(f"Early stopping at epoch {epoch}")
                break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', choices=['sage', 'gat', 'stable_gat'], default='sage',
                        help='Model type: sage=YourGNNModel, gat=GATModel, stable_gat=StableGATModel')
    parser.add_argument('--hidden', type=int, default=16, help='Hidden dimension size')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads (for GAT)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Confidence threshold for pseudo-labeling')
    parser.add_argument('--epochs', type=int, default=300, help='Total epochs')
    parser.add_argument('--es_iters', type=int, help='Early stopping patience')
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    args = parser.parse_args()

    # Device setup
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    features, graph, num_classes, train_labels_np, val_labels_np, test_labels_np, \
    train_mask_np, val_mask_np, test_mask_np = load_data()

    # Feature scaling
    features = StandardScaler().fit_transform(features.numpy())
    features = torch.tensor(features, dtype=torch.float32)

    # Convert labels and masks to torch tensors
    train_labels = torch.tensor(train_labels_np, dtype=torch.long)
    val_labels = torch.tensor(val_labels_np, dtype=torch.long)
    test_labels = torch.tensor(test_labels_np, dtype=torch.long)
    train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_np, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

    # Move data to device
    graph = graph.to(device)
    features = features.to(device)
    train_labels = train_labels.to(device)
    val_labels = val_labels.to(device)
    test_labels = test_labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    in_size = features.shape[1]
    out_size = num_classes

    # Helper to build model based on args
    def build_model():
        if args.model == 'sage':
            return YourGNNModel(in_size, args.hidden, out_size).to(device)
        if args.model == 'gat':
            return GATModel(in_size, args.hidden, out_size, num_heads=args.num_heads).to(device)
        if args.model == 'stable_gat':
            return StableGATModel(in_size, args.hidden, out_size, num_heads=args.num_heads).to(device)

    # 1) Initial training
    model = build_model()
    print("Starting initial training...")
    train(graph, features, train_labels, val_labels, train_mask, val_mask,
          model, args.epochs, args.es_iters)

    # 2) Pseudo-labeling
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)
    pseudo_mask = (confidences > args.threshold) & test_mask
    pseudo_labels = preds[pseudo_mask]
    print(f"Pseudo-labeling {pseudo_mask.sum().item()} nodes (threshold={args.threshold})")

        # 3) Combine labels and mask
    # Create full-length label vector
    num_nodes = features.shape[0]
    full_labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
    full_labels[train_mask] = train_labels
    full_labels[pseudo_mask] = pseudo_labels
    combined_mask = train_mask | pseudo_mask
    # Extract labels for combined training set
    retrain_labels = full_labels[combined_mask]

    # 4) Retraining with pseudo-labels
    model = build_model()
    print("Retraining with pseudo-labeled data...")
    train(graph, features, retrain_labels, val_labels, combined_mask, val_mask,
          model, args.epochs, args.es_iters)

    # 5) Final testing and export
print("Final testing and exporting predictions...")
model.eval()
with torch.no_grad():
    logits = model(graph, features)
    logits = logits[test_mask]
    _, test_preds = torch.max(logits, dim=1)

out_path = 'output.csv'
with open(out_path, 'w') as f:
    f.write('ID,Predict\n')
    for idx, pred in enumerate(test_preds):
        f.write(f'{idx},{int(pred)}\n')
print(f"Done. Predictions saved to {out_path}")
