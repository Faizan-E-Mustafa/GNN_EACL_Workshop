import ijson
import json
import src.config


def split_raw_data(data_path, save_path, filter_keys, split_size):

    f = open(data_path)

    o = ijson.items(f, "articles.item")
    # g = (i['articles'] for i in o )
    data = []
    split_no = 0
    splits_marks = set(range(split_size, 16300000, split_size))
    print(f"Splits = {splits_marks}")
    for i, j in enumerate(o):
        # print(v)
        j = {k: v for k, v in j.items() if k in ["pmid", "meshMajor"]}
        data.append(j)
        if i in splits_marks:

            out = {"articles": data}
            with open(save_path / f"split{split_no}.json", "w") as outfile:
                json.dump(out, outfile)
            print(f"Split No = {split_no} saved ..")
            split_no = split_no + 1
            data = []
            del out

    # last
    out = {"articles": data}
    with open(save_path / f"split{split_no}.json", "w") as outfile:
        json.dump(out, outfile)


(src.config.DATA_PREPROCESSED / "splits").mkdir(parents=True, exist_ok=True)
split_raw_data(
    data_path=src.config.DATA_RAW / "allMeSH_2022.json",
    save_path=src.config.DATA_PREPROCESSED / "splits",
    filter_keys=["pmid", "meshMajor"],
    split_size=400000,
)
