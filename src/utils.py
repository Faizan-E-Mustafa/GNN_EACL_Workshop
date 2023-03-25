import json
import src.config

import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA


def convert_heading_id_to_name():
    with open(
        src.config.DATA_ROOT
        / "test/results"
        / "MTIDEF_Results_testset_Batch1_Week01_Jan17.json",
        "r",
        encoding="utf-8",
    ) as f:
        test_dataset = json.load(f)["documents"]

    with open(src.config.DATA_PREPROCESSED / "pmid_headingname_map.json", "r") as f:
        pmid_headingname_map = json.load(f)

    # headingname_pmid_map = {j:i for i, j in pmid_headingname_map.items()}
    result = []
    # import pdb; pdb.set_trace()
    for doc_id, article_headings in enumerate(test_dataset):
        temp = []
        for i in article_headings["labels"]:
            try:
                name = pmid_headingname_map[i]
                temp.append(name)
            except:
                continue

        result.append({doc_id: temp})
    result_dict = {"documents": result}

    with open(
        src.config.DATA_ROOT
        / "test/results"
        / "MTIDEF_Results_testset_Batch1_Week01_Jan17_names.json",
        "w",
    ) as f:
        json.dump(result_dict, f, indent=2)


import numpy as np


def ramdomly_filter_articles(article_embeddigs, article_labels, percentage):
    idxs = np.random.choice(
        range(len(article_embeddigs)),
        int(len(article_embeddigs) * percentage),
        replace=False,
    )
    article_embeddigs = article_embeddigs[idxs, :]
    idx = set(idxs)
    article_labels = [j for i, j in enumerate(article_labels) if i in idx]
    return article_embeddigs, article_labels


def visualize(h, c, save_path):
    # from sklearn.manifold import TSNE
    
    
    z = PCA(n_components=2).fit_transform(h)
    # z = umap.UMAP().fit_transform(h)

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=3, c=c, alpha=0.2)
    # import pdb; pdb.set_trace()
    
    
    plt.show()
    plt.savefig(save_path)
    plt.close()
  

if __name__ == "__main__":
    convert_heading_id_to_name()
