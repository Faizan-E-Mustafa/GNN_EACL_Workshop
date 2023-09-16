
# GNN Link Prediction 

This repository contains the code for the following [paper](https://aclanthology.org/2023.insights-1.9/)
: Annotating PubMed Abstracts with MeSH Headings using Graph Neural Network 

## Abstract:


*The number of scientific publications in the biomedical domain is continuously increasing with time. An efficient system for indexing these publications is required to make the information accessible according to the user’s information needs. Task 10a of the BioASQ challenge aims to classify PubMed articles according to the MeSH ontology so that new publications can be grouped with similar preexisting publications in the field without the assistance of time-consuming and costly annotations by human annotators. In this work, we use Graph Neural Network (GNN) in the link prediction setting to exploit potential graph-structured information present in the dataset which could otherwise be neglected by transformer-based models. Additionally, we provide error analysis and a plausible reason for the substandard performance achieved by GNN.*

## Setup 

Create new virtual environment if necessary

```
python -m venv .venv
```
Python version: 3.10


Once environment is activated use following command to install required packages.

```bash
pip install -e .
``` 

```bash
pip install -r requirements.txt
```

`data` folder can be downloaded from [here](https://drive.google.com/drive/folders/17HQiKmJEW8L3wetQO-BL8eWLU3LxNdf_?usp=share_link)

It should contain following files:

![folder structure](data_tree_structure.PNG)

## Preprocessing

Following command prepares required embeddings.

```bash
python src/preprocessing.py
```

Following command prepares datasets and creates negative edges before training so that the same edges can be used for different runs. 

```bash
python src/graph_preparation.py
```

## Training 

Train GNN model. BCE or Focal Loss

```bash
python gnn.py
```

Train GNN with Dynamic Random sampling

```bash
python gnn_drs.py
```

Train GNN with mixup

```bash
python gnn_mixup.py
```

### BibTeX
```json
@inproceedings{mustafa-etal-2023-annotating,
    title = "Annotating {P}ub{M}ed Abstracts with {M}e{SH} Headings using Graph Neural Network",
    author = "Mustafa, Faizan E  and
      Boutalbi, Rafika  and
      Iurshina, Anastasiia",
    booktitle = "The Fourth Workshop on Insights from Negative Results in NLP",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.insights-1.9",
    doi = "10.18653/v1/2023.insights-1.9",
    pages = "75--81",
}
```

