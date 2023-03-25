import json
import pickle
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np


class FilterMeSHHeadings:
    def __init__(self, xml_file_path, splitted_filpath, groupname_savepath):
        self.splitted_filpath = splitted_filpath
        self.groupname_savepath = groupname_savepath
        tree = ET.parse(xml_file_path)
        self.root = tree.getroot()

    def select_headings_at_depth(self, depth=8):
        "A Mesh heading can have more than 1 tree numbers which may have different depths. We select mesh headings at depth 8"

        headings_at_depth_n = set()

        for child in self.root.findall("./DescriptorRecord"):

            for g_child in child.iter("TreeNumberList"):
                treenumber_counts = [
                    i
                    for i in g_child.findall("./TreeNumber")
                    if len(i.text.split(".")) == depth
                ]
                if len(treenumber_counts) > 0:
                    try:
                        heading_descriptor = child.find("./DescriptorName/String").text
                    except:
                        import pdb

                        pdb.set_trace()
                    headings_at_depth_n.add(heading_descriptor)
        print(f"Heading at depth {depth} = {len(headings_at_depth_n)}")

        return headings_at_depth_n

    def get_group_for_article(
        self, article_headings, pmid, headings_at_depth_n, groupnames
    ):
        "Make a group for an article based on the headings_at_depth_n"

        article_headings = [i for i in article_headings if i in headings_at_depth_n]
        #         import pdb; pdb.set_trace()

        if article_headings:
            article_headings = sorted(article_headings, key=str.lower)
            groupname = "_".join(article_headings)

            if groupname not in groupnames:
                groupnames.add(groupname)
        else:
            groupname = ""

        return pmid, groupname, groupnames

    def get_groups_for_all_articles(self, headings_at_depth_n, no_splits):
        "Get all the groups for all articles based on the headings_at_depth_n"
        for split_no in range(no_splits):
            with open(self.splitted_filpath / f"split{split_no}.json", "r") as f:
                data = json.load(f)["articles"]

            headings = [i["meshMajor"] for i in data]
            ids = [i["pmid"] for i in data]
            #             import pdb; pdb.set_trace()

            article_group = {}
            if split_no == 0:
                groupnames = set()

            for article_headings, pmid in tqdm(zip(headings, ids), total=len(headings)):
                id_, groupname, groupnames = self.get_group_for_article(
                    article_headings, pmid, headings_at_depth_n, groupnames
                )
                article_group[id_] = groupname
            print(
                f"No. of groups after processing {split_no} splits = {len(groupnames)}"
            )
            with open(
                self.groupname_savepath / f"article_assingned{split_no}.json", "w"
            ) as outfile:
                json.dump(article_group, outfile)

    def headings_above_depth(self, depth=8, save_path = ''):
        "MeSH headings above and including a certain depth"

        headings_above_depth_n = set()

        for child in tqdm(self.root.findall("./DescriptorRecord")):

            for g_child in child.iter("TreeNumberList"):
                treenumber_counts = [
                    i
                    for i in g_child.findall("./TreeNumber")
                    if len(i.text.split(".")) <= depth
                ]
                if len(treenumber_counts) > 0:
                    try:
                        heading_descriptor = child.find("./DescriptorName/String").text
                    except:
                        import pdb

                        pdb.set_trace()
                    headings_above_depth_n.add(heading_descriptor)
        print(f"Heading above and including depth {depth} = {len(headings_above_depth_n)}")
        if save_path:
            with open(save_path, "wb") as fOut:
                pickle.dump({"headings_above_depth_n": headings_above_depth_n}, fOut)


        return headings_above_depth_n 


from sentence_transformers import SentenceTransformer


class Embedding:
    def __init__(self, model_name="all-distilroberta-v1") -> None:
        self.model = SentenceTransformer(model_name, cache_folder=".")
        # pass

    def get_heading_metadata_for_embedding(
        self, xml_file_path, meta_data_save_path=None
    ):
        tree = ET.parse(xml_file_path)
        self.root = tree.getroot()

        from collections import OrderedDict

        heading_meta_data = OrderedDict()

        for child in self.root.findall("./DescriptorRecord"):
            heading_descriptor = child.find("./DescriptorName/String").text
            scope_notes = child.findall("./ConceptList/Concept/ScopeNote")
            if len(scope_notes) == 1:
                scope_note = scope_notes[0].text
            elif len(scope_notes) == 2:
                # TODO: some concepts have two scope notes. which scope note to select?
                scope_note = scope_notes[0].text
            elif len(scope_notes) == 0:
                # TODO: Currently neglect headings with 0 notes
                # import pdb; pdb.set_trace()
                continue

            heading_meta_data[heading_descriptor] = {"scope_note": scope_note}

        if meta_data_save_path:
            with open(meta_data_save_path, "w") as outfile:
                json.dump(heading_meta_data, outfile)
        return heading_meta_data

    def encode(self, text, save_path=None):

        embeddings = self.model.encode(
            text,
            show_progress_bar=True,
            device="cuda",
            batch_size=128
            #    output_value='token_embeddings'
        )
        print(embeddings.shape)
        if save_path:
            with open(save_path, "wb") as fOut:
                pickle.dump({"embeddings": embeddings}, fOut)
        return embeddings

    def concat(embed_tuple):
        return np.vstack(embed_tuple)


from sentence_transformers import util
import torch


class SimilarEmbedding(Embedding):
    def __init__(self, model_name):
        super().__init__(model_name)

    def find_similar(self, query, documents):
        documents_embeddings = self.model.encode(documents, convert_to_tensor=True)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(5, len(documents))
        for query in queries:
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, documents_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")

            for score, idx in zip(top_results[0], top_results[1]):
                print(documents[idx], "(Score: {:.4f})".format(score))

    def find_similar_using_embeddings(
        self, query_embeddings, document_embeddings, n, return_score=False
    ):

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = min(n, len(document_embeddings))
        results = []
        for query_embedding in tqdm(query_embeddings):

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = util.cos_sim(query_embedding, document_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)
            if return_score:
                results.append((top_results[0], top_results[1]))
            else:
                results.append(top_results[1])
        return results

    def documents_idx_to_headings(self, documents_idx, article_headings):
        # import pdb; pdb.set_trace()
        return [
            [article_headings[idx] for idx in doc_idx.tolist()]
            for doc_idx in documents_idx
        ]

    def score_headings_in_similar_documents(
        self, similar_documents_headings, similarity_scores
    ):
        "For a query, get score for the headings in similar retieved documents"
        total_similarity_score = [sum(i.tolist()) for i in similarity_scores]
        # import pdb; pdb.set_trace()
        unique_headings_in_similar_doc = set(
            [k for i in similar_documents_headings for j in i for k in j]
        )

        unique_heading_scores_querys = []

        for query_id, query_similar_docs in tqdm(enumerate(similar_documents_headings)):
            unique_heading_scores_query = {}
            for selected_heading in unique_headings_in_similar_doc:
                selected_heading_score = 0
                for doc_id, doc_headings in enumerate(query_similar_docs):

                    doc_headings = set(doc_headings)

                    if selected_heading in doc_headings:
                        # import pdb; pdb.set_trace()
                        selected_heading_score += similarity_scores[query_id][
                            doc_id
                        ].item()

                selected_heading_score = (
                    selected_heading_score / total_similarity_score[query_id]
                )
                selected_heading_score = round(selected_heading_score, 2)
                if selected_heading_score:
                    unique_heading_scores_query[
                        selected_heading
                    ] = selected_heading_score

            unique_heading_scores_query = {
                k: v
                for k, v in sorted(
                    unique_heading_scores_query.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
            unique_heading_scores_querys.append(unique_heading_scores_query)
        # import pdb; pdb.set_trace()
        return unique_heading_scores_querys

    def generate_search_space(self, unique_heading_scores_querys, savepath=""):
        result = [list(i.keys()) for i in unique_heading_scores_querys]
        if savepath:
            with open(savepath, "wb") as outfile:
                pickle.dump(result, outfile)
        return result

    # def score_headings_single_query(
    #     self, similar_documents_headings, similarity_scores
    # ):


import src.config

if __name__ == "__main__":
    emb = Embedding()
    # heading_meta_data = emb.get_heading_metadata_for_embedding(
    #     xml_file_path=src.config.DATA_RAW / "desc2022.xml",
    #     meta_data_save_path=src.config.DATA_PREPROCESSED / "heading_meta_data.json",
    # )

    # headings_text = [i["scope_note"] for i in heading_meta_data.values()]
    # emb.encode(
    #     headings_text, save_path=src.config.DATA_PREPROCESSED / "heading_embedding.pkl"
    # )

    # with open(
    #     src.config.DATA_ROOT / "test" / "Task10a-Batch1-Week1_raw.json",
    #     "r",
    #     encoding="utf-8",
    # ) as f:
    #     test_dataset = json.load(f)["documents"]
    # test_article_abstracts = [i["abstractText"] for i in test_dataset]
    # emb.encode(
    #     test_article_abstracts,
    #     save_path=src.config.DATA_PREPROCESSED / "test_embedding.pkl",
    # )

    import pdb; pdb.set_trace()
    with open(
            src.config.DATA_PREPROCESSED / "filtered_articles.json", "r"
        ) as outfile:
            filtered_articles = json.load(outfile)["articles"]
    emb.encode(
        filtered_articles,
        save_path=src.config.DATA_PREPROCESSED / "article_embedding.pkl",
    )
    

    
    # filter_headings = FilterMeSHHeadings(xml_file_path = src.config.DATA_RAW / 'desc2022.xml',
    #                                 splitted_filpath = src.config.DATA_PREPROCESSED / "splits",
    #                                 groupname_savepath = src.config.DATA_PREPROCESSED / "splits"
    #                                 )
    # headings_above_depth_n = filter_headings.headings_above_depth(depth=2, save_path=src.config.DATA_PREPROCESSED /'headings_above_depth_n.pkl')