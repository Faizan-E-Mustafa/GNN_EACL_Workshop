import os.path as osp
from collections import Counter
import random


import torch
from sklearn.metrics import roc_auc_score
import numpy as np

from torch import nn
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
import torch_geometric

import src.config
import src.graph_preparation
from src.utils import visualize
import logging
import os

logging.basicConfig(filename="logfile_focal.log", filemode="w", level=logging.DEBUG)

logger = logging.getLogger()

# device_num = os.environ['CUDA_VISIBLE_DEVICES']
# print(n)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
# torch.cuda.set_device(1)
train_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="train_data_without_neg.pt"
)[0]
val_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="val_data_without_neg.pt"
)[0]
# import pdb; pdb.set_trace()
mix_gcf = src.graph_preparation.MixGCF()



class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # import pdb; pdb.set_trace()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.2, training=self.training)
        return self.conv2(x, edge_index)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma=2,
        alpha=0.25,
    ):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([1 - alpha, alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # import pdb; pdb.set_trace()
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at *  (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


model = Net(768, 256, 128).to(device)
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=0.005,
    #  weight_decay=5e-4
)
criterion = torch.nn.BCEWithLogitsLoss(
#     #  pos_weight=torch.tensor([0.7]).to(device)
)
# criterion = FocalLoss(gamma=2, alpha=0.4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)


def train(epoch):
    model.train()
    optimizer.zero_grad()
    enc_embs = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
    
    # import pdb; pdb.set_trace()
    if epoch < -1:
        select = epoch % 99

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = np.load(
            src.config.GNN_DATA_ROOT
            / "processed/edge_label_index_15_reduced_headings"
            / f"edge_label_index_{select}.npy"
        )
        neg_edge_index = torch.tensor(neg_edge_index).T
        # import pdb; pdb.set_trace()

        edge_label_index = torch.cat(
            [train_data.edge_label_index, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat(
        [
            train_data.edge_label,
            train_data.edge_label.new_zeros(neg_edge_index.size(1)),
        ],
        dim=0,
    )
        out = model.decode(enc_embs, edge_label_index).view(-1)

    else: 
        enc_embs = enc_embs.unsqueeze(1)

        user_gcn_emb, item_gcn_emb= enc_embs[:mix_gcf.n_train, :], enc_embs[mix_gcf.n_train:, :]
    
        neg_edge_index = mix_gcf.get_negative_edges(
                mix_gcf._article_assigned_labels,
            ) # (No of train articles, 15)
        users = train_data.edge_label_index[0,:]
        items = mix_gcf._train_to_heading_idx(train_data.edge_label_index[1,:].tolist())
        items = torch.tensor(items)
        neg_edge_index_mix_gcf, neg_indices= mix_gcf.mix_gcf_negative_sampling(user_gcn_emb, item_gcn_emb,
        neg_edge_index,users,items, 0.9)

        #filtering the negative which is selected by mixgcf
        neg_indices = np.choose(neg_indices.flatten().cpu(), neg_edge_index[users].T)
        neg_indice = mix_gcf._heading_to_train_idx(neg_indices)
        heading_emb_neg = enc_embs[neg_indice]
       
        edge_label = torch.cat(
            [
                train_data.edge_label,
                train_data.edge_label.new_zeros(train_data.edge_label.size(0)),
            ],
            dim=0,
        )
        # 
        article_emb_pos = enc_embs[train_data.edge_label_index[0]]
        heading_emb_pos = enc_embs[train_data.edge_label_index[1]]
        if epoch%2 == 0:
            idx = random.sample(range(article_emb_pos.shape[0]), 10000)
            t = ['y' for i in range(10000)] + ['b' for i in range(10000)]  + ['r' for i in range(10000)]
            emb = torch.cat((article_emb_pos[idx], heading_emb_neg[idx], neg_edge_index_mix_gcf[idx]), 0).squeeze(1).cpu().detach().numpy() 
            # import pdb; pdb.set_trace()
            visualize(emb , t, f'debugging/plots/epoch_{epoch}')

        out_pos = (article_emb_pos * heading_emb_pos).sum(dim=-1)
        out_neg = (article_emb_pos * neg_edge_index_mix_gcf).sum(dim=-1)
        out = torch.cat((out_pos, out_neg), 0).flatten()
    # import pdb; pdb.set_trace()
    loss = criterion(out.to(device), edge_label.to(device))
    loss.backward()
    optimizer.step()
    return loss
     
    # edge_label_index, edge_label


@torch.no_grad()
def test(x, edge_index, edge_label_index, edge_label, epoch, mode):
    model.eval()
    z = model.encode(x.to(device), edge_index.to(device))
    out_prob = model.decode(z, edge_label_index).view(-1).sigmoid()
    # import pdb; pdb.set_trace()

    out = torch.where(out_prob > 0.5, 1, 0)
    cm = confusion_matrix(edge_label.flatten().cpu().numpy(), out.cpu().numpy())

    
    print(cm)
    return f1_score(edge_label.flatten().cpu().numpy(), out.cpu().numpy())


best_val_auc = 0
for epoch in range(0, 300):

    loss = train(epoch)

    val_auc = test(
        val_data.x,
        val_data.edge_index,
        val_data.edge_label_index,
        val_data.edge_label,
        epoch,
        "val",
    )
    # train_auc = test(
    #     train_data.x,
    #     train_data.edge_index,
    #     edge_label_index,
    #     edge_label,
    #     epoch,
    #     "train",
    # )
    scheduler.step()
    if val_auc > best_val_auc:
        best_val = val_auc

    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {val_auc:.4f}, Val: {val_auc:.4f}, "
    )