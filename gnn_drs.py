import os.path as osp
from typing import OrderedDict
from collections import Counter
import random

import torch
from sklearn.metrics import roc_auc_score
import numpy as np

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import negative_sampling
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import torch_geometric
from torch import nn

import src.config
import src.graph_preparation
import logging

logging.basicConfig(filename="debugging/gnn_onfly.log", filemode="w", level=logging.DEBUG)

logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

train_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="train_data_without_neg.pt"
)[0]
val_data = src.graph_preparation.MyOwnDataset(
    root="./data_gnn", ds=None, ds_name="val_data_without_neg.pt"
)[0]

neg_samp = src.graph_preparation.NegativeSampling()

temp = []
unique_headings = neg_samp.unique_headings
unique_headings_dict = dict(zip(unique_headings, range(len(unique_headings))))
article_assigned_labels = neg_samp._article_assigned_labels
for i in article_assigned_labels:
    for j in i:
        try:
            temp.append(unique_headings_dict[j])
        except:
            continue
train_heading_freq = Counter(temp)


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
        F_loss =  at * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()



    
model = Net(768, 256, 128).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.005,
#  weight_decay=0.0005
 )
criterion = torch.nn.BCEWithLogitsLoss(
    # pos_weight=torch.tensor([0.2])
    )
# model.load_state_dict(torch.load('model.pth'))
# criterion = FocalLoss(gamma=2, alpha=0.4)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=1)


def train(epoch):
    edges_weights = None
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
    # import pdb; pdb.set_trace()
    if epoch <= 2:
        neg_edge_index = np.load(
            src.config.GNN_DATA_ROOT
            / "processed/edge_label_index_15_reduced_headings"
            / f"edge_label_index_{epoch}.npy"
        )
        neg_edge_index = torch.tensor(neg_edge_index).T
    else:
        

        neg_edge_index = train_data.neg_edge_index

        edges_weights = torch.cat(
        [
            train_data.edge_label.new_ones(train_data.edge_label_index.size(1)),
            train_data.edge_prob
        ],
        dim=0,
    )   
        edges_weights = edges_weights.to(device)
    
        

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
    
    
        
    loss = criterion(out.to(device), edge_label.to(device))
    
    # loss = F.binary_cross_entropy_with_logits(out.to(device), edge_label.to(device) , weight=edges_weights)
    loss.backward()
    optimizer.step()
    return loss, edge_label_index, edge_label

from src.utils import visualize
@torch.no_grad()
def test(x, edge_index, edge_label_index, edge_label, epoch, mode):
    model.eval()
    z = model.encode(x.to(device), edge_index.to(device))
    out_prob = model.decode(z, edge_label_index).view(-1).sigmoid()

    from collections import Counter

    out = torch.where(out_prob > 0.5, 1, 0)
    cm = confusion_matrix(edge_label.flatten().cpu().numpy(), out.cpu().numpy())

    if mode == "train":
        if epoch >= 2: 
            mask =  torch.logical_and((edge_label.to(device) == 0),( out.to(device)==1)) #fp
            incorrect_edge_label_index = edge_label_index[
                :, mask
            ]
            incorrect_prob = out_prob[mask]
            # 
            # incorrect_prob_temp, idxs = torch.sort(incorrect_prob,descending =True)
            # logging.debug(incorrect_prob_temp[:500].tolist())

            # logging.debug(Counter(incorrect_edge_label_index[0, :].tolist()).most_common())
            # logging.debug(Counter(incorrect_edge_label_index[1, :].tolist()).most_common())
            # import pdb; pdb.set_trace()
            # if epoch % 5 == 0 or
            # We perform a new round of negative sampling for every training epoch:
            #
            neg_edge_index, edge_prob = neg_samp.get_negative_edges_on_fly(
                incorrect_edge_label_index, 
                neg_samp._article_assigned_labels,
                incorrect_prob,
                lower_threshold=0.6,
                upper_threshold= 0.95
                
            )
            
            
            neg_edge_index = torch.tensor(neg_edge_index, dtype=torch.long)
            neg_edge_index = neg_edge_index.t().contiguous()
            train_data.neg_edge_index = neg_edge_index
            train_data.edge_prob = edge_prob

            # logging.debug(Counter(neg_edge_index[0, :].tolist()).most_common())
            # logging.debug(Counter(neg_edge_index[1, :].tolist()).most_common())

            # logging most incorrect headings and the frequency in the training set
            # headings = neg_edge_index[1, :].tolist()
            # val_incorrect = neg_samp._train_to_heading_idx(headings)
            # val_incorrect_freq = Counter(val_incorrect).most_common()
            # logging.info(val_incorrect_freq)
            # val_incorrect_freq_training_occurence = [
            #     train_heading_freq[i] for i, j in val_incorrect_freq
            # ]
            # logging.info(val_incorrect_freq_training_occurence)

            hard_negative_idxs = np.argwhere(np.array(edge_prob) !=1).flatten()
            # import pdb; pdb.set_trace()

            if epoch % 5 == 0:
                import random
                idx = random.sample(range(train_data.edge_label_index.shape[-1]), 20000)
                all_article_embs = z[train_data.edge_label_index[0,:]][idx]
                all_heading_embs = z[train_data.edge_label_index[1,:]][idx]
                hard_heading_embs = z[neg_edge_index[1,hard_negative_idxs]]
                t = ['y' for i in range(20000)] + ['b' for i in range(20000)] + ['r' for i in range(hard_heading_embs.shape[0])]
                emb = torch.cat((all_article_embs, all_heading_embs, hard_heading_embs), 0).squeeze(1).cpu().detach().numpy()
                
                visualize(emb , t, f'debugging/plots_onfly/epoch_{epoch}')
            # import pdb; pdb.set_trace()
    # if mode != "train":
    # #     # if cm[0, 1] < 60000 and cm[1, 1] > 900:
    #     if epoch > 115:
    #         incorrect = edge_label_index[
    #             :,
    #             torch.logical_and(
    #                 edge_label != out.cpu(), edge_label == 0
    #             ).flatten(),
    #         ]
    #         headings = incorrect[1, :].tolist()
    #         val_incorrect = neg_samp._val_to_heading_idx(headings)
    #         val_incorrect_freq = Counter(val_incorrect).most_common()
    #         logging.info(val_incorrect_freq)
    #         val_incorrect_freq_training_occurence = [
    #             train_heading_freq[i] for i, j in val_incorrect_freq
    #         ]
    #         logging.info(val_incorrect_freq_training_occurence)
    # if epoch > 110:
    #     if cm[0, 1] < 23000 and cm[1, 1] > 900:
    #         torch.save(model.state_dict(), 'model.pth')
    #         raise Exception
    print(cm)
    return f1_score(edge_label.flatten().cpu().numpy(), out.cpu().numpy())


best_val_auc = 0
for epoch in range(0, 500):

    loss, edge_label_index, edge_label = train(epoch)

    train_auc = test(train_data.x, train_data.edge_index, edge_label_index, edge_label, epoch, 'train')
    val_auc = test(
        val_data.x,
        val_data.edge_index,
        val_data.edge_label_index,
        val_data.edge_label,
        epoch,
        "val",
    )
    scheduler.step()
    if val_auc > best_val_auc:
        best_val = val_auc

    print(
        f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {val_auc:.4f}, Val: {val_auc:.4f}, "
    )


# z = model.encode(test_data.x, test_data.edge_index)
# final_edge_index = model.decode_all(z)
