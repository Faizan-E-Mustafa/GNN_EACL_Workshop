import numpy as np
import pickle
import json

import src.config
import src.graph_preparation


class LTR:
    def __init__(self) -> None:
        self._load_artifacts()
        headings = self.heading_meta_data.keys()
        graph_prep = src.graph_preparation.GraphPreparation()
        unique_headings = graph_prep.unique_headings
        self.all_unique_headings = dict(
            zip(unique_headings, range(len(unique_headings)))
        )

    def create_dataset(self, knn_search_space_train, correct_headings, save_path):
        assert len(knn_search_space_train) == (len(correct_headings))
        queries_info = []
        import pdb

        pdb.set_trace()
        for idx, search_space in enumerate(knn_search_space_train):

            correct_heading = correct_headings[idx]
            correct_heading = set(correct_heading)
            relevance_score = []
            filtered_docs_idx = []
            for h in search_space:
                if h in self.all_unique_headings:  # some headings are being ignored
                    filtered_docs_idx.append(self.all_unique_headings[h])

                    if h in correct_heading:
                        relevance_score.append(1)
                    else:
                        relevance_score.append(0)

            print(f"{str(sum(relevance_score))}, {len(correct_heading)}")
            filtered_docs = self.heading_embedding[filtered_docs_idx, :]
            # filtered_docs = np.random.normal( size=(len(filtered_docs_idx), 128))

            query_id = np.full((len(filtered_docs), 1), idx)

            relevance_score = np.array(relevance_score).reshape(-1, 1)
            query_info = np.hstack((query_id, filtered_docs, relevance_score))
            queries_info.append(query_info)

        dataset = np.vstack(tuple(queries_info))
        if save_path:
            np.save(save_path, dataset)

    def _load_artifacts(self):

        val_graph = np.load(src.config.GNN_DATA_ROOT / "outputs/val_graph_embs.npy")
        self.heading_embedding = val_graph[50100:]

        with open(
            src.config.DATA_PREPROCESSED / "heading_meta_data.json", "r"
        ) as outfile:
            self.heading_meta_data = json.load(outfile)

        with open(
            src.config.DATA_PREPROCESSED / "ltr/knn_search_space_train.pkl", "rb"
        ) as fOut:
            self.knn_search_space_train = pickle.load(fOut)


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

    test_article_labels = [
        [pmid_headingname_map.get(l, "") for l in labels]
        for labels in test_article_labels
    ][
        100:
    ]  # change this

    ltr = LTR()
    knn_search_space_train = ltr.knn_search_space_train
    ltr.create_dataset(
        knn_search_space_train,
        test_article_labels,
        save_path=src.config.DATA_PREPROCESSED / "ltr" / "ltr_train_dataset_gnn.npy",
    )
