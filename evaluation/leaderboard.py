# from huggingface_hub import snapshot_download
# snapshot_download(repo_id="open-ko-llm-leaderboard/results", repo_type="dataset")
import json
import re
from glob import glob

import pandas as pd
from tqdm import tqdm

col_order = [
    "avg",
    "ko_gpqa_diamond_zeroshot",
    "ko_winogrande",
    "ko_gsm8k",
    "ko_eqbench",
    "ko_ifeval",
    "kornat_common",
    "kornat_social",
    "kornat_harmless",
    "kornat_helpful",
]

model_list = ["llama", "qwen", "gemma", "mistral", "yi", "solar", "eeve", "phi"]

root_dir = "/Users/songhak/.cache/huggingface/hub/datasets--open-ko-llm-leaderboard--results/snapshots/00876f4fe27009e634f630c59bd7aba51b0ca0fe"
fp_list = glob(f"{root_dir}/*/*/*.json")

result = pd.DataFrame()
for fp in tqdm(fp_list):
    data = json.load(open(fp))
    name = data["config_general"]["model_name"].lower()
    df = pd.DataFrame(data["results"])
    # pd.DataFrame(data["results"])["ko_ifeval"]
    df = df.loc[
        [
            index
            for index in df.index
            if all(
                c not in index
                for c in ["err", "alias", "loose", "exact_match,flexible-extract", "percent_parseable,none"]
            )
        ]
    ]
    displayed = [df[col].dropna().item() if col != "ko_ifeval" else df[col].dropna().mean() for col in df.columns]

    displayed = pd.DataFrame([displayed], index=[name], columns=df.columns)
    displayed[(displayed < 1) & (displayed > 0)] *= 100
    displayed["avg"] = displayed.mean(axis=1).item()
    displayed = displayed[col_order].round(2)

    saved_type = None
    for n in model_list:
        if n in name:
            saved_type = n
            break

    displayed["model_type"] = saved_type
    parameter = re.findall(r"\d{1,3}b", name)
    if parameter:
        displayed["size"] = int(parameter[0][:-1])

    result = result._append(displayed)

result = result.sort_values("avg", ascending=False)

multiindex = list(map(lambda x: [x.split("/", maxsplit=1)[0], x], result.index))
multiindex = pd.MultiIndex.from_arrays(list(zip(*multiindex)))
result.index = multiindex

target = result[(result["size"] < 10) | (result["size"].isna())].drop(columns="size")
target = target.groupby("model_type").describe().round(2)
target = target.unstack()[[(i, j, k) for i, j, k in target.unstack().index if j in ["mean", "max"]]].unstack()


for col in target.T.columns.get_level_values(0).unique():
    print(col)
    print(target.loc[col].sort_values("max", axis=1, ascending=False))


# for col in target.select_dtypes(np.number):
#     # index = target[col].sort_values(ascending=False).iloc[:top_k].index
#     target.groupby("model_type").mean()
#
#     print(f"{col}")
#     top_models = target.loc[index]
#     print("\t",top_models[col].mean().round(), top_models["model_type"].dropna().unique())
