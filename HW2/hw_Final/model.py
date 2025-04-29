# grace_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv,GATv2Conv
from dgl.nn.pytorch import SAGEConv    

class GCN_LPA(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, dropout, lpa_iters=10, lpa_weight=0.5):
        super(GCN_LPA, self).__init__()
        self.gc1 = GraphConv(in_feats, hidden_feats)
        self.gc2 = GraphConv(hidden_feats, out_feats)
        self.dropout = dropout
        self.lpa_iters = lpa_iters
        self.lpa_weight = lpa_weight

    def forward(self, g, features, labels, mask):
        # GCN部分
        x = F.relu(self.gc1(g, features))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gc2(g, x)

        # 手動建立adjacency matrix
        src, dst = g.edges()
        adj = torch.sparse_coo_tensor(
            torch.stack([src, dst]),
            torch.ones(src.shape[0], device=src.device),
            (g.number_of_nodes(), g.number_of_nodes())
        )

        # LPA部分
        y = labels.clone()
        y[~mask] = 0
        for _ in range(self.lpa_iters):
            y = torch.sparse.mm(adj, y)
            y[mask] = labels[mask]

        return x, y

class GATv2Encoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 num_heads1=8, num_heads2=4, dropout=0.5):
        super().__init__()
        # 每層輸出維度 = 每頭 out_feats × num_heads
        self.gat1 = GATv2Conv(in_dim,  hid_dim // num_heads1,
                              num_heads1,
                              feat_drop=dropout, attn_drop=dropout,
                              activation=F.elu,
                              allow_zero_in_degree=True)
        self.gat2 = GATv2Conv(hid_dim, out_dim // num_heads2,
                              num_heads2,
                              feat_drop=dropout, attn_drop=dropout,
                              activation=None,
                              allow_zero_in_degree=True)

        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)

    def forward(self, g, x):
        h = self.gat1(g, x).flatten(1)   # (N, hid_dim)
        h = self.bn1(h);  h = F.elu(h)

        h = self.gat2(g, h).flatten(1)     # 2nd 層用 mean 合併 heads → (N, out_dim)
        h = self.bn2(h)
        return h

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super().__init__()
        # -------- GraphConv --------
        # allow_zero_in_degree=True 可避免 0-in-degree 警告
        self.conv1 = GraphConv(in_dim, hid_dim,
                               activation=None,
                               allow_zero_in_degree=True)
        self.conv2 = GraphConv(hid_dim, out_dim,
                               activation=None,
                               allow_zero_in_degree=True)

        # -------- BatchNorm (放在每層 conv 後) --------
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        # self.norm1 = nn.LayerNorm(hid_dim)   # ← 就是放這裡
        # self.norm2 = nn.LayerNorm(out_dim)   # ← 以及這裡
        

        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, g, x):
        # --- 第一層 ---
        h1 = self.conv1(g, x)
        h1 = self.bn1(h1)
        #h1 = self.norm1(h1)
        h1 = self.act(h1)          # 先 BN 再 ReLU
        h1 = self.drop(h1)

        # --- 第二層 ---
        h2 = self.conv2(g, h1)
        h2 = self.bn2(h2)
        #h2 = self.norm2(h2)

        # --- 殘差 (輸入與輸出同維度即可相加) ---
        # 改進殘差連接 - 確保維度匹配
        if h1.shape[-1] != h2.shape[-1]:
            h1 = nn.Linear(h1.shape[-1], h2.shape[-1])(h1)
        h = h2 + h1
        h = self.act(h)
        h = self.drop(h)
        return h

class GRACE(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, g1, x1, g2, x2):
        z1 = self.encoder(g1, x1)
        z2 = self.encoder(g2, x2)
        p1 = self.project(z1)
        p2 = self.project(z2)
        return p1, p2

class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, agg_type='mean', dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim, aggregator_type=agg_type)
        self.conv2 = SAGEConv(hid_dim, out_dim, aggregator_type=agg_type)
        self.bn1 = nn.BatchNorm1d(hid_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # 殘差連接改為跨層跳躍 (hid_dim → out_dim)
        self.skip = nn.Sequential(
            nn.Linear(hid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
    def forward(self, g, x):
        # 第一層
        h1 = self.conv1(g, x)
        h1 = self.bn1(h1)
        h1 = self.act(h1)
        h1 = self.drop(h1)
        
        # 第二層
        h2 = self.conv2(g, h1)
        h2 = self.bn2(h2)
        
        # 殘差連接：h1 → skip → 與 h2 相加
        h = h2 + self.skip(h1)  # 確保維度匹配
        h = self.act(h)
        h = self.drop(h)
        return h


    def __init__(self, args, features, labels, adj):
        self.args = args
        self.vars = []  # for computing l2 loss

        self._build_inputs(features, labels)
        self._build_edges(adj)
        self._build_gcn(features[2][1], labels.shape[1], features[0].shape[0])
        self._build_lpa()
        self._build_train()
        self._build_eval()

    def _build_inputs(self, features, labels):
        self.features = tf.SparseTensor(*features)
        self.labels = tf.constant(labels, dtype=tf.float64)
        self.label_mask = tf.placeholder(tf.float64, shape=labels.shape[0])
        self.dropout = tf.placeholder(tf.float64)

    def _build_edges(self, adj):
        edge_weights = glorot(shape=[adj[0].shape[0]])
        self.adj = tf.SparseTensor(adj[0], edge_weights, adj[2])
        self.normalized_adj = tf.sparse_softmax(self.adj)
        self.vars.append(edge_weights)

    def _build_gcn(self, feature_dim, label_dim, feature_nnz):
        hidden_list = []

        if self.args.gcn_layer == 1:
            gcn_layer = GCNLayer(input_dim=feature_dim, output_dim=label_dim, adj=self.normalized_adj,
                                 dropout=self.dropout, sparse=True, feature_nnz=feature_nnz, act=lambda x: x)
            self.outputs = gcn_layer(self.features)
            self.vars.extend(gcn_layer.vars)
        else:
            gcn_layer = GCNLayer(input_dim=feature_dim, output_dim=self.args.dim, adj=self.normalized_adj,
                                 dropout=self.dropout, sparse=True, feature_nnz=feature_nnz)
            hidden = gcn_layer(self.features)
            hidden_list.append(hidden)
            self.vars.extend(gcn_layer.vars)

            for _ in range(self.args.gcn_layer - 2):
                gcn_layer = GCNLayer(input_dim=self.args.dim, output_dim=self.args.dim, adj=self.normalized_adj,
                                     dropout=self.dropout)
                hidden = gcn_layer(hidden_list[-1])
                hidden_list.append(hidden)
                self.vars.extend(gcn_layer.vars)

            gcn_layer = GCNLayer(input_dim=self.args.dim, output_dim=label_dim, adj=self.normalized_adj,
                                 dropout=self.dropout, act=lambda x: x)
            self.outputs = gcn_layer(hidden_list[-1])
            self.vars.extend(gcn_layer.vars)

        self.prediction = tf.nn.softmax(self.outputs, axis=-1)

    def _build_lpa(self):
        label_mask = tf.expand_dims(self.label_mask, -1)
        input_labels = label_mask * self.labels
        label_list = [input_labels]

        for _ in range(self.args.lpa_iter):
            lp_layer = LPALayer(adj=self.normalized_adj)
            hidden = lp_layer(label_list[-1])
            label_list.append(hidden)
        self.predicted_label = label_list[-1]

    def _build_train(self):
        # GCN loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.labels)
        self.loss = tf.reduce_sum(self.loss * self.label_mask) / tf.reduce_sum(self.label_mask)

        # LPA loss
        lpa_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predicted_label, labels=self.labels)
        lpa_loss = tf.reduce_sum(lpa_loss * self.label_mask) / tf.reduce_sum(self.label_mask)
        self.loss += self.args.lpa_weight * lpa_loss

        # L2 loss
        for var in self.vars:
            self.loss += self.args.l2_weight * tf.nn.l2_loss(var)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr).minimize(self.loss)

    def _build_eval(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float64)
        self.accuracy = tf.reduce_sum(correct_prediction * self.label_mask) / tf.reduce_sum(self.label_mask)