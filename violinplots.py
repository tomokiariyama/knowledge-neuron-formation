import os
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.graphs import make_whole_concepts_histogram


def get_data(result_path, model_steps):
    dfs_suppress_relevant = []
    dfs_suppress_unrelated = []
    dfs_enhance_relevant = []
    dfs_enhance_unrelated = []

    for step in model_steps:
        # Path
        suppress_relevant = os.path.join(result_path, f"multiberts-seed_0-step_{step}k", "nt_4_at_0.2_mw_10",
                                         "suppress_activation_and_relevant_prompts.jsonl"
                                         )
        suppress_unrelated = os.path.join(result_path, f"multiberts-seed_0-step_{step}k", "nt_4_at_0.2_mw_10",
                                          "suppress_activation_and_unrelated_prompts.jsonl"
                                          )
        enhance_relevant = os.path.join(result_path, f"multiberts-seed_0-step_{step}k", "nt_4_at_0.2_mw_10",
                                        "enhance_activation_and_relevant_prompts.jsonl"
                                        )
        enhance_unrelated = os.path.join(result_path, f"multiberts-seed_0-step_{step}k", "nt_4_at_0.2_mw_10",
                                         "enhance_activation_and_unrelated_prompts.jsonl"
                                         )

        df_suppress_relevant, df_suppress_unrelated = make_whole_concepts_histogram(suppress_relevant,
                                                                                    suppress_unrelated,
                                                                                    graph_type="suppress",
                                                                                    only_return_dataframes=True
                                                                                    )
        df_enhance_relevant, df_enhance_unrelated = make_whole_concepts_histogram(enhance_relevant,
                                                                                  enhance_unrelated,
                                                                                  graph_type="enhance",
                                                                                  only_return_dataframes=True
                                                                                  )

        # 適宜DataFrameの内容を編集・追記
        df_suppress_relevant["学習ステップ数"] = step
        df_suppress_unrelated["学習ステップ数"] = step
        df_enhance_relevant["学習ステップ数"] = step
        df_enhance_unrelated["学習ステップ数"] = step

        df_suppress_relevant["凡例"] = "正例文を予測"
        df_suppress_unrelated["凡例"] = "負例文を予測"
        df_enhance_relevant["凡例"] = "正例文を予測"
        df_enhance_unrelated["凡例"] = "負例文を予測"

        # violinplotをするにあたり，便宜上+100%を超える確率の変化量を持つ概念を対象外とする
        df_suppress_relevant = df_suppress_relevant[df_suppress_relevant["正解を選ぶ確率の変化率[%]"] <= 100]
        df_suppress_unrelated = df_suppress_unrelated[df_suppress_unrelated["正解を選ぶ確率の変化率[%]"] <= 100]
        df_enhance_relevant = df_enhance_relevant[df_enhance_relevant["正解を選ぶ確率の変化率[%]"] <= 100]
        df_enhance_unrelated = df_enhance_unrelated[df_enhance_unrelated["正解を選ぶ確率の変化率[%]"] <= 100]

        dfs_suppress_relevant.append(df_suppress_relevant)
        dfs_suppress_unrelated.append(df_suppress_unrelated)
        dfs_enhance_relevant.append(df_enhance_relevant)
        dfs_enhance_unrelated.append(df_enhance_unrelated)

    dfs = {
        "dfs_sr": dfs_suppress_relevant,
        "dfs_su": dfs_suppress_unrelated,
        "dfs_er": dfs_enhance_relevant,
        "dfs_eu": dfs_enhance_unrelated
        }

    return dfs


def make_parser():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "-rp", "--result_path",
        help="path for the result file (e.g.) file at 'work/result/generics_kb_best')",
        required=True
        )
    parser.add_argument(
        '--save_path',
        help="results(contain used entities) will be saved under the designed path.",
        type=str, default='work/figure/generics_kb_best'
        )

    return parser.parse_args()


def main():
    args = make_parser()

    activation_edit_types = ["suppress", "enhance"]
    steps = [0, 20, 40, 60, 80, 100, 200, 300, 400, 500, 1000, 1500, 2000]

    # データの取得・変形
    raw_data = get_data(args.result_path, steps)

    for df in (raw_data["dfs_sr"] + raw_data["dfs_su"]):
        try:
            df_suppress = pd.concat([df_suppress, df])
        except NameError:
            df_suppress = df

    for df in (raw_data["dfs_er"] + raw_data["dfs_eu"]):
        try:
            df_enhance = pd.concat([df_enhance, df])
        except NameError:
            df_enhance = df

    # グラフの描画
    fontsize = 12
    for df, edit_type in zip([df_suppress, df_enhance], activation_edit_types):
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))

        sns.violinplot(data=df, x="学習ステップ数", y="正解を選ぶ確率の変化率[%]", hue="凡例",
                       split=True, scale="count", cut=0, ax=ax
                       )

        ax.set_xlabel("チェックポイントの学習ステップ数[×10^3]", fontsize=fontsize)
        ax.set_ylabel("正解確率の相対変化[%]", fontsize=fontsize)
        if edit_type == "enhance":
            ax.legend(loc=4, fontsize=fontsize)  # 凡例の位置調整
        else:
            ax.legend(loc=1, fontsize=fontsize)

        figure = fig.get_figure()

        os.makedirs(args.save_path, exist_ok=True)
        save_path = os.path.join(args.save_path, f"violinplot_{edit_type}.png")
        figure.savefig(save_path)
        print(f"figure is saved in {save_path}")
        plt.close()


if __name__ == '__main__':
    main()
