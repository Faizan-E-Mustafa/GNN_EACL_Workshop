from collections import defaultdict
import json
from math import ceil
import pickle
import src.config
import src.utils
import numpy as np
from pathlib import Path

# np.random.seed(42)

from tqdm import tqdm
from torch_geometric.data import Data
import torch
from torch import nn


class GraphPreparation:
    def __init__(self) -> None:
        self.n_borrowed_graph = 40000
        self.n_train = 70000
        self.n_test = 100
        self._load_artifacts()
        # self.n_train = len(self.article_headings)
        # import pdb; pdb.set_trace()

        self._article_assigned_labels = [i["meshMajor"] for i in self.filtered_articles]
        temp_unique_headings = list(self.heading_meta_data.keys())
        # self.unique_headings = temp_unique_headings

        # filtering headings based on training article headings
        train_article_headings = self._unique_headings_in_articles(
            self._article_assigned_labels
        )
        # import pdb; pdb.set_trace()
        self.unique_headings = [
            i for i in temp_unique_headings if i in train_article_headings
        ]
        # import pdb; pdb.set_trace()
        self.unique_headings_index = [
            temp_unique_headings.index(i) for i in self.unique_headings
        ]
        self.heading_embedding = self.heading_embedding[self.unique_headings_index]
        # import pdb; pdb.set_trace()

    def _unique_headings_in_articles(self, article_assigned_labels):
        return {heading for i in article_assigned_labels for heading in i}

    def get_train_graph(self, add_negative_edges):

        article_assigned_labels = self._article_assigned_labels

        assert len(self.unique_headings) == self.heading_embedding.shape[0]
        assert len(article_assigned_labels) == self.article_embedding.shape[0]
        # article_assigned_labels = article_assigned_labels[:100]

        embeddings = np.vstack((self.article_embedding, self.heading_embedding))

        edge_index = self.get_edge_index(
            article_assigned_labels,
            self.unique_headings,
        )

        edge_index = np.array(edge_index)
        np.random.shuffle(edge_index)

        edge_indexs = np.array_split(edge_index, [int(edge_index.shape[0] * 0.6)], 0)
        mp_edge_index = edge_indexs[0].tolist()
        pos_edge_label_index = edge_indexs[1].tolist()
        # import pdb; pdb.set_trace()

        if add_negative_edges:  # adding negative edges one time
            # alternative is to use self.get_negative_edges to get multiple random negative edges

            neg_edge_label_index = self.get_negative_edges(
                article_assigned_labels,
                self.unique_headings,
            )

        x = torch.tensor(embeddings, dtype=torch.float)
        y = torch.rand((1, len(embeddings)))

        mp_edge_index = torch.tensor(mp_edge_index, dtype=torch.long)
        mp_edge_index = mp_edge_index.t().contiguous()

        pos_edge_label_index_labels = torch.ones((len(pos_edge_label_index),))

        if add_negative_edges:
            neg_edge_label_index_labels = torch.zeros(
                (
                    len(
                        neg_edge_label_index,
                    )
                )
            )

            edge_label = torch.cat(
                (pos_edge_label_index_labels, neg_edge_label_index_labels), 0
            )
            edge_label_index = pos_edge_label_index + neg_edge_label_index
        else:
            edge_label_index = pos_edge_label_index

            edge_label = pos_edge_label_index_labels

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_label_index = edge_label_index.t().contiguous()

        train_data = Data(x=x, edge_index=mp_edge_index, y=y)
        train_data.edge_label_index = edge_label_index
        train_data.edge_label = edge_label

        # train_data.sp_edges = edge_label_index
        # train_data.gold_edges = edge_label
        # import pdb; pdb.set_trace()
        return train_data

    def get_negative_edges(
        self,
        article_assigned_labels,
        unique_headings,
    ):

        n_articles = len(article_assigned_labels)
        n_headings = len(unique_headings)

        edge_index = []

        max_links = 15
        for article_index, heading_labels in tqdm(
            enumerate(article_assigned_labels), total=n_articles
        ):
            heading_labels = set(heading_labels)
            # article_headings =set([unique_headings.index(heading) for heading in heading_labels])

            counter = 0
            possible_headings = random.sample(range(n_headings), 30)
            # possible_headings = np.random.choice(n_headings, 30, replace=False)
            for heading_index in possible_headings:
                heading_label = unique_headings[heading_index]
                #
                if heading_label not in heading_labels:
                    heading_index = heading_index + n_articles
                    edge_index.append([article_index, heading_index])
                    counter = counter + 1
                else:
                    # print(f"{heading_label} not found in article {article_index}")
                    continue

                if counter == max_links:

                    break

        return edge_index

    def _val_to_train_heading_idx(self, incorrect_headings_from_val):
        return [
            self.n_train + (i - (self.n_borrowed_graph + self.n_test))
            for i in incorrect_headings_from_val
        ]

    def _val_to_heading_idx(self, incorrect_headings_from_val):
        return [
            i - (self.n_borrowed_graph + self.n_test)
            for i in incorrect_headings_from_val
        ]

    def _train_to_heading_idx(self, incorrect_headings_from_train):
        return [i - (self.n_train) for i in incorrect_headings_from_train]
    
    def _heading_to_train_idx(self, heading_idxs):
        return [ self.n_train + i for i in heading_idxs]

    def get_negative_edges_on_fly(
        self,
        article_assigned_labels,  # train
        unique_headings,
        incorrect_headings_train_idx,  # decending
        incorrect_freq,  # decending
        mode,
    ):
        "Negative edges based on frequency obtained from the wrong predicted negative edges"

        n_articles = len(article_assigned_labels)
        if mode == "train":
            incorrect_headings_idx = self._train_to_heading_idx(
                incorrect_headings_train_idx
            )  # original unique heading index
        elif mode == "val":
            incorrect_headings_idx = self._val_to_heading_idx(
                incorrect_headings_train_idx
            )
        incorrect_headings_idx = incorrect_headings_idx[
            :100
        ]  # + incorrect_headings_idx[-100:]
        incorrect_freq = incorrect_freq[:100]  # + incorrect_freq[:100]

        incorrect_freq = incorrect_freq / np.sum(incorrect_freq)
        incorrect_freq = torch.tensor(incorrect_freq)
        incorrect_headings_idx = torch.tensor(incorrect_headings_idx)
        edge_index = []
        # import pdb; pdb.set_trace()
        for article_index, heading_labels in tqdm(
            enumerate(article_assigned_labels), total=n_articles
        ):
            heading_labels = set(heading_labels)
            # article_headings =set([unique_headings.index(heading) for heading in heading_labels])

            idx = incorrect_freq.multinomial(num_samples=6, replacement=True)
            possible_headings = incorrect_headings_idx[idx]
            # possible_headings = np.random.choice(incorrect_headings_idx, 20, p=inccorrect_freq)
            for heading_index in possible_headings:

                heading_label = unique_headings[heading_index]

                if heading_label not in heading_labels:
                    heading_index = heading_index + n_articles
                    edge_index.append([article_index, heading_index])

                else:
                    # print(f"{heading_label} not found in article {article_index}")
                    continue

        return edge_index

    def get_edge_index(
        self,
        article_assigned_labels,
        unique_headings,
    ):
        "get edges between article and heading nodes"
        n_articles = len(article_assigned_labels)

        edge_index = []

        for article_index, heading_labels in tqdm(
            enumerate(article_assigned_labels), total=n_articles
        ):
            for heading in heading_labels:
                try:
                    heading_index = unique_headings.index(heading)
                except:  # some headings do not have scope note. They are neglected
                    continue
                heading_index = heading_index + n_articles

                edge_index.append([article_index, heading_index])

        return edge_index

    def _load_artifacts(
        self,
    ):

        with open(
            src.config.DATA_PREPROCESSED / "filtered_articles.json", "r"
        ) as outfile:
            self.filtered_articles = json.load(outfile)["articles"][: self.n_train]
        # with open(src.config.DATA_PREPROCESSED / "article_labels_year.pkl", "rb") as fOut:
        #     article_headings = pickle.load(fOut)["article_labels"]

        with open(
            src.config.DATA_PREPROCESSED / "heading_meta_data.json", "r"
        ) as outfile:
            self.heading_meta_data = json.load(outfile)

        with open(src.config.DATA_PREPROCESSED / "article_embedding.pkl", "rb") as fOut:
            self.article_embedding = pickle.load(fOut)["embeddings"][: self.n_train]

        # with open(src.config.DATA_PREPROCESSED / "article_embedding_year.pkl", "rb") as fOut:
        #     article_embedding = pickle.load(fOut)["embeddings"]

        with open(src.config.DATA_PREPROCESSED / "heading_embedding.pkl", "rb") as fOut:
            self.heading_embedding = pickle.load(fOut)["embeddings"]

        # self.article_embedding, self.article_headings = src.utils.ramdomly_filter_articles(article_embedding,
        #  article_headings,
        #  percentage=0.15)

    def get_test_graph(self, test_embeddings_path, article_assigned_labels):
        with open(test_embeddings_path, "rb") as fOut:
            test_article_embeddings = pickle.load(fOut)["embeddings"]

        test_article_embeddings = test_article_embeddings[: self.n_test]
        article_assigned_labels = article_assigned_labels[: self.n_test]

        n_headings = self.heading_embedding.shape[0]
        n_test_article = test_article_embeddings.shape[0]

        # n_borrowed_graph = len(self.article_embedding)

        edge_index, borrowed_embeddings = self.get_borrowed_graph(
            n_article_nodes=self.n_borrowed_graph, n_test_article=n_test_article
        )
        embeddings = np.vstack(
            (borrowed_embeddings, test_article_embeddings, self.heading_embedding)
        )
        edge_label_index = self.get_all_possibe_edge_index(
            self.n_borrowed_graph,
            n_test_article,
            n_headings,
        )

        # edge_label_index = self.get_edges_based_on_knn_search_space(self.n_borrowed_graph,
        #     n_test_article,)
        # import pdb; pdb.set_trace()
        edge_label = self.get_edge_label(
            article_assigned_labels, self.n_borrowed_graph, edge_label_index
        )
        edge_label = torch.tensor(edge_label, dtype=torch.float).reshape(1, -1)

        x = torch.tensor(embeddings, dtype=torch.float)
        y = torch.rand((1, len(embeddings)))

        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous()

        edge_label_index = torch.tensor(edge_label_index, dtype=torch.long)
        edge_label_index = edge_label_index.t().contiguous()

        test_data = Data(x=x, edge_index=edge_index, y=edge_label)
        test_data.edge_label_index = edge_label_index
        test_data.edge_label = edge_label
        # import pdb; pdb.set_trace()
        # test_data.sp_edges = edge_label_index
        # test_data.gold_edges = edge_label
        return test_data
        # Task10a-Batch1-Week1_raw.json

    def get_edges_based_on_knn_search_space(
        self,
        n_borrowed_graph,
        n_test_article,
    ):
        with open(src.config.DATA_PREPROCESSED / "knn_search_space.pkl", "rb") as f:
            knn_search_space_queries = pickle.load(f)

        # test articles are queries and train are documents
        assert len(knn_search_space_queries) == n_test_article
        edge_label_index = []
        for query_id, query_search_space in enumerate(knn_search_space_queries):
            query_search_space_idx = []
            for i in query_search_space:
                try:
                    query_search_space_idx.append(self.unique_headings.index(i))
                except:
                    continue
            test_node_id = n_borrowed_graph + query_id
            heading_node_id = n_borrowed_graph + n_test_article
            edge_label_index.extend(
                [[test_node_id, heading_node_id + i] for i in query_search_space_idx]
            )
        return edge_label_index

    def get_all_possibe_edge_index(
        self,
        n_borrowed_graph,
        n_test_article,
        n_headings,
    ):
        # skip message-passing edges for borrowed graph
        # import pdb; pdb.set_trace()
        t = n_borrowed_graph + n_test_article
        # TODO: recheck
        return [
            [i, j]
            for i in tqdm(range(n_borrowed_graph, t))
            for j in range(t, t + n_headings)
        ]

    def get_borrowed_graph(self, n_article_nodes, n_test_article):

        borrowed_articles_idx = np.random.choice(
            len(self.article_embedding), n_article_nodes, replace=False
        )

        # borrowed_articles_idx = range(len(self.article_embedding))

        borrowed_articles_idx = set(borrowed_articles_idx)

        borrowed_articles = [
            x
            for i, x in enumerate(self.filtered_articles)
            if i in borrowed_articles_idx
        ]

        article_assigned_labels = [i["meshMajor"] for i in borrowed_articles]
        # article_assigned_labels = [
        #     x
        #     for i, x in enumerate(self.article_headings)
        #     if i in borrowed_articles_idx
        # ]

        borrowed_articles_idx = sorted(list(borrowed_articles_idx))
        borrowed_embeddings = self.article_embedding[borrowed_articles_idx, :]

        # for correct edge index , make supervision edges for borrowd graph only
        article_assigned_labels = article_assigned_labels + [
            [] for i in range(n_test_article)
        ]
        edge_index = self.get_edge_index(
            article_assigned_labels,
            self.unique_headings,
        )
        return edge_index, borrowed_embeddings

    def get_edge_label(
        self, article_assigned_labels, n_borrowed_graph, edge_label_index
    ):

        article_assigned_labels = [
            [] for i in range(n_borrowed_graph)
        ] + article_assigned_labels

        pos_edge_index = self.get_edge_index(
            article_assigned_labels,
            self.unique_headings,
        )

        pos_edge_index = set([(i, j) for i, j in pos_edge_index])
        edge_label = [1 if (i, j) in pos_edge_index else 0 for i, j in edge_label_index]
        return edge_label

    # def get_edge_label_knn(
    #     self, article_assigned_labels, n_borrowed_graph, edge_label_index
    # ):
    #     'get labels for edges for knn search space'
    #     article_assigned_labels = [
    #         [] for i in range(n_borrowed_graph)
    #     ] + article_assigned_labels

    #     pos_edge_index = self.get_edge_index(
    #         article_assigned_labels,
    #         self.unique_headings,
    #     )

    #     pos_edge_index = set([(i, j) for i, j in pos_edge_index])
    #     edge_label = [1 if (i, j) in pos_edge_index else 0 for i, j in edge_label_index]
    #     return edge_label

    def prepare_negative_indexes(self, n, article_assigned_labels, save_path=""):
        for i in tqdm(range(n)):
            edge_label_index = self.get_negative_edges(
                article_assigned_labels,
                self.unique_headings,
            )
            Path(f'{save_path}').mkdir(parents=True, exist_ok=True)
            if save_path:
                np.save(
                    save_path / f"edge_label_index_{i}.npy", np.array(edge_label_index)
                )
                # with open(save_path / f'edge_label_index_{i}.pkl', "wb") as fOut:
                #     pickle.dump(fOut, edge_label_index)
        return edge_label_index


import random


class NegativeSampling(GraphPreparation):
    def __init__(self) -> None:
        super().__init__()

        self.unique_headings_dict = dict(
            zip(
                range(len(self.unique_headings)),
                self.unique_headings,
            )
        )

    def get_negative_edges_on_fly(
        self,
        incorrect_edge_label_index,
        train_article_labels,
        incorrect_prob,
        lower_threshold=0.65,  # selecting hard examples
        upper_threshold=0.8,  # for ignoring really hard examples which are same as articles
    ):
        # import pdb; pdb.set_trace()
        n_articles = len(train_article_labels)
        n_headings = len(self.unique_headings_dict)

        incorrect_prob, sort_indices = torch.sort(incorrect_prob, descending=True)
        incorrect_edge_label_index = incorrect_edge_label_index[:, sort_indices]

        incorrect_edge_label_index = incorrect_edge_label_index[
            :,
            torch.logical_and(
                incorrect_prob > lower_threshold, incorrect_prob < upper_threshold
            ),
        ]

        print(f"Total hard negatives = {incorrect_edge_label_index.shape[-1]}  ")

        edge_index = []
        edge_prob = [] # weights for loss function
        
        max_hard_links = 10  # per article
        max_random_links = 5  # per article
        edge_prob_random_links = [1]*max_random_links

        incorrect_edge_label_index = incorrect_edge_label_index.cpu().detach().numpy()
        
        for article_index, article_headings in tqdm(
            enumerate(train_article_labels), total=n_articles
        ):
            n_from_incorrect = 0
            article_headings = set(article_headings)

            # add incorrect edges for a article
            filter_idxs = np.argwhere(incorrect_edge_label_index[0, :] == article_index).flatten()
            filtered_edge_index = incorrect_edge_label_index[
                :, filter_idxs
            ]
           
            filtered_incorrect_prob = incorrect_prob[filter_idxs]
            
            if len(filtered_edge_index.flatten()) != 0:
                # import pdb; pdb.set_trace()
                filtered_edge_index_temp = filtered_edge_index.T.tolist()

                filtered_edge_index_temp = filtered_edge_index_temp[:max_hard_links]
                incorrect_prob_temp = filtered_incorrect_prob[:max_hard_links] + 10 # extra weight for hard nagatives 
                incorrect_prob_temp = incorrect_prob_temp.tolist() 

                edge_index.extend(filtered_edge_index_temp)
                edge_prob.extend(incorrect_prob_temp)

                n_from_incorrect = len(filtered_edge_index_temp)
                if n_from_incorrect > max_hard_links:
                    print(
                        f"Hard edges exceeded than {max_hard_links} for article {article_index}"
                    )

            # Add remaining edges randomly until max_edges

            counter = 0

            edge_prob.extend(edge_prob_random_links) # No special treatment for random edges

            # Uniform weights for random draw

            possible_headings = random.sample(
                range(n_headings), max_random_links + ceil((max_random_links + 2 / 2))
            )

            # possible_headings = np.random.choice(n_headings, 15, replace=False)
            for heading_index in possible_headings:
                heading_label = self.unique_headings_dict[
                    heading_index
                ]  # unique_id:heading

                if heading_label not in article_headings:
                    heading_index = (
                        heading_index + n_articles
                    )  # heading position in train graph
                    edge_index.append([article_index, heading_index])
                    
                    counter = counter + 1
                else:
                    # print(f"{heading_label} not found in article {article_index}")
                    continue

                if counter == max_random_links:

                    break
        try:
            assert (len(edge_index)) <= (article_index + 1) * (
                max_hard_links + max_random_links
            )
        except:
            import pdb

            pdb.set_trace()
            print("h")


        edge_prob = torch.tensor(edge_prob, dtype=torch.float32) 
        return edge_index, edge_prob


class MixGCF(NegativeSampling):
    def __init__(self) -> None:
        super().__init__()

    def mix_gcf_negative_sampling(
        self, user_gcn_emb, item_gcn_emb, neg_candidates, users, items, pos_influence
    ):
        batch_size = users.shape[0]
        s_e, p_e = (
            user_gcn_emb[users],
            item_gcn_emb[items],
        )  # [batch_size, n_hops+1, channel]
        # if self.pool != "concat":
        #     s_e = self.pooling(s_e).unsqueeze(dim=1)
        # import pdb; pdb.set_trace()
        neg_candidates = neg_candidates[users]

        """positive mixing"""
        # neg_candidates = neg_candidates[:, 0].flatten()
        # # seed = torch.rand(batch_size,1,  p_e.shape[1]).to(p_e.device)  # (0, 1)
        # seed = torch.full((batch_size, 1, 1), pos_influence).to(p_e.device)
        # # seed = seed[None, None]
        # n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        # # n_e_ = seed * p_e + (1 - seed) * n_e  # mixing
        # n_e_ = seed * p_e + (1 - seed) * s_e  # mixing
        # return n_e_

        # import pdb; pdb.set_trace()
        seed = torch.full((batch_size, 1, p_e.shape[1], 1), pos_influence).to(p_e.device)
        # seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(
            dim=-1
        )  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        # indices = torch.topk(scores, dim=1, k = 3)[1][:, -1 , :].detach() # selecting 3rd hard negative instead of first
        # import pdb; pdb.set_trace()
        neg_items_emb_ = n_e_.permute(
            [0, 2, 1, 3]
        )  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[
            [[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :
        ], indices

    def get_negative_edges(
        self,
        article_assigned_labels,
    ):

        n_articles = len(article_assigned_labels)
        n_headings = len(self.unique_headings_dict)

        neg_headings = []  # [[candidate negative headings for first article],[]]

        max_links = 5
        for article_index, heading_labels in tqdm(
            enumerate(article_assigned_labels), total=n_articles
        ):
            heading_labels = set(heading_labels)
            neg_candidates_temp = []

            counter = 0
            possible_headings = random.sample(range(n_headings), 30)
            # possible_headings = np.random.choice(n_headings, 30, replace=False)
            for heading_index in possible_headings:
                heading_label = self.unique_headings_dict[
                    heading_index
                ]  # unique_id:heading

                if heading_label not in heading_labels:
                    heading_index = heading_index + n_articles
                    neg_candidates_temp.append(heading_index)

                    counter = counter + 1
                else:
                    # print(f"{heading_label} not found in article {article_index}")
                    continue

                if counter == max_links:
                    # import pdb; pdb.set_trace()
                    neg_candidates_temp = self._train_to_heading_idx(
                        neg_candidates_temp
                    )
                    neg_headings.append(neg_candidates_temp)

                    break
        neg_headings = np.vstack(neg_headings)
        return neg_headings


from torch_geometric.data import Dataset
import os.path as osp


class MyOwnDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        ds=None,
        ds_name=None,
    ):
        self.ds = ds
        self.ds_name = ds_name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [
            self.ds_name,
        ]

    @property
    def processed_file_names(self):
        return [
            self.ds_name,
        ]

    def download(self):
        # Download to `self.raw_dir`.
        pass

    #     ...

    def process(self):
        torch.save(self.ds, osp.join(self.processed_dir, self.ds_name))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.ds_name))
        return data


if __name__ == "__main__":

    with open(
        src.config.DATA_ROOT
        / "test/results"
        / "MTIDEF_Results_testset_Batch1_Week01_Jan17.json",
        "r",
        encoding="utf-8",
    ) as f:
        test_dataset = json.load(f)["documents"]
    test_article_labels = [i["labels"] for i in test_dataset]

    with open(src.config.DATA_PREPROCESSED / "pmid_headingname_map.json", "r") as f:
        pmid_headingname_map = json.load(f)
    # import pdb; pdb.set_trace()
    test_article_labels = [
        [pmid_headingname_map.get(l, "") for l in labels]
        for labels in test_article_labels
    ]

    graph_prep = GraphPreparation()
    article_assigned_labels = graph_prep._article_assigned_labels
    
    # uncomment for the first time only 
    edge_label_index1 = graph_prep.prepare_negative_indexes(
        100,
        article_assigned_labels,
        save_path=src.config.GNN_DATA_ROOT
        / "processed/edge_label_index_15_reduced_headings",
    )
    #######################
    import pdb; pdb.set_trace()
    train_data = graph_prep.get_train_graph(add_negative_edges=False)

    test_data = graph_prep.get_test_graph(
        test_embeddings_path=src.config.DATA_PREPROCESSED / "test_embedding.pkl",
        article_assigned_labels=test_article_labels,
    )

    G1_ds = MyOwnDataset(
        root="./data_gnn", ds=train_data, ds_name="train_data_without_neg.pt"
    )
    G2_ds = MyOwnDataset(
        root="./data_gnn",
        ds=test_data,
        ds_name="test_data_without_neg.pt",
    )


    with open(src.config.DATA_PREPROCESSED / "article_embedding.pkl", "rb") as fOut:
        article_embedding = pickle.load(fOut)["embeddings"][100000: 100100 ]

    import pickle
    with open(src.config.DATA_PREPROCESSED / 'val_embedding.pkl', 'wb') as f:
        pickle.dump({'embeddings': article_embedding}, f)

    with open(
            src.config.DATA_PREPROCESSED / "filtered_articles.json", "r"
        ) as outfile:
            filtered_articles = json.load(outfile)["articles"][100000: 100100 ]

    val_assigned_labels = [i["meshMajor"] for i in filtered_articles]

    val_data = graph_prep.get_test_graph(
        test_embeddings_path=src.config.DATA_PREPROCESSED / "val_embedding.pkl",
        article_assigned_labels=val_assigned_labels,
    )

    
    G3_ds = MyOwnDataset(
        root="./data_gnn",
        ds=val_data,
        ds_name="val_data_without_neg.pt",
    )
