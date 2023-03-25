import os.path as osp
from collections import Counter

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
import logging

logging.basicConfig(filename="logfile_focal.log", filemode="w", level=logging.DEBUG)

logger = logging.getLogger()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

train_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="train_data_without_neg.pt"
)[0]
val_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="test_data_without_neg.pt"
)[0]

test_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="val_data_without_neg.pt"
)[0]
# import pdb; pdb.set_trace()
graph_prep = src.graph_preparation.GraphPreparation()
temp = []
unique_headings = graph_prep.unique_headings
unique_headings_dict = dict(zip(unique_headings, range(len(unique_headings))))
article_assigned_labels = graph_prep._article_assigned_labels
for i in article_assigned_labels:
    for j in i:
        try:
            temp.append(unique_headings_dict[j])
        except:
            continue
train_heading_freq = Counter(temp)
#
# train_data.x = torch.tensor(StandardScaler().fit_transform(train_data.x), dtype= torch.float32)
# val_data.x = torch.tensor(StandardScaler().fit_transform(val_data.x), dtype= torch.float32)
class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.conv3 = SAGEConv(hidden_channels*2, hidden_channels)
        # self.conv4 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def encode(self, x, edge_index):
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        # x = F.dropout(x, p=0.2, training=self.training)
        temp_edge_index = dropout_adj(edge_index, p=0.3, training=self.training)[0]
        # import pdb; pdb.set_trace()
        x = self.conv1(x, temp_edge_index).relu()

        # x = self.conv3(x, dropout_adj(edge_index, p= 0.3, training=self.training)[0]).relu()
        # x = self.conv4(x, dropout_adj(edge_index, p= 0.3, training=self.training)[0]).relu()

        temp_edge_index = dropout_adj(edge_index, p=0.3, training=self.training)[0]
        x = self.conv2(x, temp_edge_index)
        # x = F.dropout(x, p=0.2, training=self.training)

        return x

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    # def decode(self, z, edge_label_index):
    #     return torch.nn.CosineSimilarity(dim=-1)(z[edge_label_index[0]], z[edge_label_index[1]])

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
        self.alpha = torch.tensor([1 - alpha, alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # import pdb; pdb.set_trace()
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss =  at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


model = Net(768, 256, 128).to(device)
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=0.005,
    #  weight_decay=5e-4
)

### Select loss
criterion = torch.nn.BCEWithLogitsLoss(
)
# criterion = FocalLoss(gamma=2, alpha=0.2)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)


def train(epoch):
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
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

    out = model.decode(z, edge_label_index).view(-1)
    # import pdb; pdb.set_trace()
    loss = criterion(out.to(device), edge_label.to(device))
    loss.backward()
    optimizer.step()
    return loss, edge_label_index, edge_label


@torch.no_grad()
def test(x, edge_index, edge_label_index, edge_label, epoch, mode):
    model.eval()
    z = model.encode(x.to(device), edge_index.to(device))
    out_prob = model.decode(z, edge_label_index).view(-1).sigmoid()
    # import pdb; pdb.set_trace()

    out = torch.where(out_prob > 0.5, 1, 0)
    # import pdb; pdb.set_trace()
    cm = confusion_matrix(edge_label.flatten().cpu().numpy(), out.cpu().numpy())

    # if mode != "train":
    #     if cm[0, 1] < 2000 and cm[1, 1] > 950:
    #         import pdb

    #         pdb.set_trace()

    if mode != "train":
        if cm[0, 1] < 55000 and cm[1, 1] > 900:
            incorrect = edge_label_index[
                :, torch.logical_and(edge_label != out.cpu(), edge_label == 0).flatten()
            ]
            headings = incorrect[1, :].tolist()
            val_incorrect = graph_prep._val_to_heading_idx(headings)
            val_incorrect_freq = Counter(val_incorrect).most_common()
            logging.info(val_incorrect_freq)
            val_incorrect_freq_training_occurence = [
                train_heading_freq[i] for i, j in val_incorrect_freq
            ]
            logging.info(val_incorrect_freq_training_occurence)

    print(cm)
    return f1_score(edge_label.flatten().cpu().numpy(), out.cpu().numpy())


best_val_auc = 0
for epoch in range(0, 100):

    loss, edge_label_index, edge_label = train(epoch)

    # val_auc = test(
    #     val_data.x,
    #     val_data.edge_index,
    #     val_data.edge_label_index,
    #     val_data.edge_label,
    #     epoch,
    #     "val",
    # )
    train_auc = test(
        train_data.x,
        train_data.edge_index,
        edge_label_index,
        edge_label,
        epoch,
        "train",
    )
    scheduler.step()
    if val_auc > best_val_auc:
        best_val = val_auc

    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {val_auc:.4f}, Val: {val_auc:.4f}, "
    )
    if epoch > 80: 
        test_auc = test(
        test_data.x,
        test_data.edge_index,
        test_data.edge_label_index,
        test_data.edge_label,
        epoch,
        "val",
    )


# z = model.encode(test_data.x, test_data.edge_index)
# final_edge_index = model.decode_all(z)
