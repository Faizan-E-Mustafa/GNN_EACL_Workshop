import json

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import src.config


with open(
    src.config.DATA_ROOT / "test/submitted" / "Task10a-Batch1-Week1_submitted.json",
    "r",
) as f:
    submission_data = json.load(f)["documents"]

with open(
    src.config.DATA_ROOT
    / "test/results/MTIDEF_Results_testset_Batch1_Week01_Jan17.json",
    "r",
) as f:
    gold_data = json.load(f)["documents"]
import pdb

# pdb.set_trace()
submitted_labels = [article["labels"] for article in submission_data]
gold_labels = [article["labels"] for article in gold_data]

mlb = MultiLabelBinarizer()
gold_labels = mlb.fit_transform(gold_labels)
submitted_labels = mlb.transform(submitted_labels)
f1 = f1_score(gold_labels, submitted_labels, average="micro")
p = precision_score(gold_labels, submitted_labels,average='micro')
r = recall_score(gold_labels, submitted_labels, average='micro')

print(f'p = {p} r = {r} f1 = {f1}, ')
