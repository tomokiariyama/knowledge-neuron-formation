import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import traceback
import sys
import nltk
import jsonlines
import japanize_matplotlib


def sns_settings():
    sns.set(
        context="paper",
        style="whitegrid",
        palette=sns.color_palette("Set1", 24),
        font_scale=4,
        rc={"lines.linewidth": 6, 'grid.linestyle': '--'},
        font='IPAexGothic'  # Japanese font
        )


def sns_settings2():
    sns.set(
        context="paper",
        style="whitegrid",
        palette=sns.dark_palette("palegreen", n_colors=2, reverse=True),
        font_scale=4,
        rc={"lines.linewidth": 6, 'grid.linestyle': '--'},
        font='IPAexGothic'
        )


def make_dict_from_result_file(result_file_path):
    prob_change_dict = defaultdict(tuple)
    ng_word_set = (
        "marijuana", "grenade", "guns", "bigotry", "rifle", "revolver", "pistol", "destruction", "terrorism", "fart",
        "farting", "urinate", "urinating", "ejaculate", "orgasm", "penis", "copulate", "copulating", "flirt", "flirting",
        "sex", "reproduce", "reproducing", "fuck", "pee", "poop", "shit"
    )
    with jsonlines.open(result_file_path) as reader:
        for record in reader:
            if record["ground_truth"] not in ng_word_set:
                prob_change_dict[record["ground_truth"]] = (
                    record["number_of_refined_neurons"],
                    record["ground_truth_prob_before_modifying"],
                    record["ground_truth_prob_after_modifying"],
                    record["sentence"]
                )

    return prob_change_dict


def extract_gt_type_dict(prob_change_dict, ground_truth_type):
    if ground_truth_type == "noun":
        # 名詞と判定されたground truthのみグラフ化の対象とする
        matched_pos_list = ["NN", "NNS", "NNP", "NNPS"]
    elif ground_truth_type == "verb_adjective_adverb":
        # 形容詞・副詞・動詞と判定されたground truthのみグラフ化の対象とする
        matched_pos_list = ["JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
    elif ground_truth_type == "others":
        matched_pos_list = ["CC", "CD", "DT", "EX", "FW", "IN", "LS", "MD", "PDT", "POS", "PRP", "PRP$", "RP", "SYM", "TO", "UH", "WDT", "WP", "WP$", "WRB", ",", "."]
    else:
        print(f"Debug: ground_truth_type '{ground_truth_type}' is wrong.")
        sys.exit(1)

    particular_gt_type_dict = defaultdict(tuple)
    for ground_truth, info_tuple in prob_change_dict.items():
        no_mask_sentence = info_tuple[3].replace('[MASK]', ground_truth)
        for morph in nltk.pos_tag(nltk.word_tokenize(no_mask_sentence)):
            if morph[0] == ground_truth and morph[1] in matched_pos_list:
                particular_gt_type_dict[ground_truth] = info_tuple
                break

    return particular_gt_type_dict


def make_datalist_for_graph(relevant_prob_change_dict, unrelated_prob_change_dict, *, graph_type: str):
    percentage_list = []
    labels = []
    ground_truth_list = []

    if graph_type == "suppress":
        axis_titlehead = "抑制"
    else:
        axis_titlehead = "増幅"

    for relevant_dict_element, unrelated_dict_element in zip(relevant_prob_change_dict.items(), unrelated_prob_change_dict.items()):
        if int(relevant_dict_element[1][0]) == 0:
            continue  # 精製されたニューロンの数が0個の場合は，グラフを作成する上で必ず0%になってしまう不都合が生じるため，除外する
        else:
            # relevant promptの場合の結果の比率を計算し，dataframeを作るためのリストに代入
            percentage_list.append((float(relevant_dict_element[1][2]) - float(relevant_dict_element[1][1])) / float(relevant_dict_element[1][1]) * 100)  # 相対変化の場合
            #percentage_list.append(float(relevant_dict_element[1][2]) - float(relevant_dict_element[1][1]))  # 絶対変化の場合
            labels.append(f"活性値を{axis_titlehead}し，知識ニューロンに紐づく概念のセンテンスを予測")
            ground_truth_list.append(relevant_dict_element[0])

            # unrelated promptの場合の結果の比率を計算し，dataframeを作るためのリストに代入
            percentage_list.append((float(unrelated_dict_element[1][2]) - float(unrelated_dict_element[1][1])) / float(unrelated_dict_element[1][1]) * 100)  # 相対変化の場合
            #percentage_list.append(float(unrelated_dict_element[1][2]) - float(unrelated_dict_element[1][1]))  # 絶対変化の場合
            labels.append(f"活性値を{axis_titlehead}し，知識ニューロンとは無関係の概念のセンテンスを予測")
            ground_truth_list.append(unrelated_dict_element[0])

    return percentage_list, labels, ground_truth_list


def get_datalist(relevant_result_path, unrelated_result_path, *, graph_type, ground_truth_type="all"):
    if ground_truth_type == "all":
        relevant_prob_change_dict = make_dict_from_result_file(relevant_result_path)
        unrelated_prob_change_dict = make_dict_from_result_file(unrelated_result_path)
    else:
        relevant_prob_change_dict = extract_gt_type_dict(make_dict_from_result_file(relevant_result_path), ground_truth_type)
        unrelated_prob_change_dict = extract_gt_type_dict(make_dict_from_result_file(unrelated_result_path), ground_truth_type)

    return make_datalist_for_graph(relevant_prob_change_dict, unrelated_prob_change_dict, graph_type=graph_type)


def make_individual_concept_graph(relevant_result_path, unrelated_result_path, root_save_path, *, entity_index, display_number_of_entities, graph_type):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_result_path, unrelated_result_path, graph_type=graph_type)

    start_entity_index = entity_index
    number_of_entities_show_in_x_axis = display_number_of_entities

    if start_entity_index >= int(len(percentage_list) / 2):
        try:
            raise ValueError(f"The argument '--displayed_entity_index' is too large. Please set it equal to, or smaller than {int(len(percentage_list)/2) - 1}")
        except ValueError:
            traceback.print_exc()
            sys.exit(1)

    if len(percentage_list) >= 2*start_entity_index + 2*number_of_entities_show_in_x_axis:
        df = pd.DataFrame({"正解を選ぶ確率の変化率[%]": percentage_list[2*start_entity_index:2*start_entity_index + 2*number_of_entities_show_in_x_axis],
                           "凡例": labels[2*start_entity_index:2*start_entity_index + 2*number_of_entities_show_in_x_axis],
                           "概念": ground_truth_list[2*start_entity_index:2*start_entity_index + 2*number_of_entities_show_in_x_axis]})
    elif len(percentage_list) >= 2*number_of_entities_show_in_x_axis:
        stop_index_after_going_around = 2*start_entity_index + 2*number_of_entities_show_in_x_axis - len(percentage_list)
        df = pd.DataFrame({"正解を選ぶ確率の変化率[%]": percentage_list[2*start_entity_index:] + percentage_list[:stop_index_after_going_around],
                           "凡例": labels[2*start_entity_index:] + labels[:stop_index_after_going_around],
                           "概念": ground_truth_list[2*start_entity_index:] + ground_truth_list[:stop_index_after_going_around]})
    else:
        try:
            raise ValueError(f"The argument '-x', or '--number_of_entities_show_in_x_axis' is too large. Please set it equal to, or smaller than {int(len(percentage_list)/2)}")
        except ValueError:
            traceback.print_exc()
            sys.exit(1)

    sns_settings()
    fig, ax = plt.subplots(1, 1, figsize=(40, 15), tight_layout=True)

    if graph_type == "suppress":
        graph = sns.barplot(x="概念", y='正解を選ぶ確率の変化率[%]', data=df, hue="凡例", ax=ax,
                            palette=sns.light_palette("blue", n_colors=3, reverse=True))
    else:
        graph = sns.barplot(x="概念", y='正解を選ぶ確率の変化率[%]', data=df, hue="凡例", ax=ax,
                            palette=sns.light_palette("red", n_colors=3, reverse=True))

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=32)  # 凡例の位置を調整する
    graph.set_xticklabels(graph.get_xticklabels(), rotation=-60)  # 概念名の表示を斜めにする
    figure = graph.get_figure()

    save_path = os.path.join(root_save_path, f"individual_{graph_type}_graph.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()


# ヒストグラム
def make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, *, graph_type, prompt_type):
    if prompt_type == "relevant":
        matched_tail_string = "紐づく概念のセンテンスを予測"
    else:
        matched_tail_string = "無関係の概念のセンテンスを予測"

    prompt_percentage_list = []
    prompt_labels = []
    prompt_ground_truth_list = []
    for percentage, label, ground_truth in zip(percentage_list, labels, ground_truth_list):
        if label.endswith(matched_tail_string):
            prompt_percentage_list.append(percentage)
            prompt_labels.append(label)
            prompt_ground_truth_list.append(ground_truth)

    df = pd.DataFrame({"正解を選ぶ確率の変化率[%]": prompt_percentage_list,
                       "凡例": prompt_labels,
                       "概念": prompt_ground_truth_list}
                      )

    return df


def make_whole_concepts_histogram(relevant_result_path, unrelated_result_path, *, root_save_path="", graph_type, only_return_dataframes=False):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_result_path, unrelated_result_path, graph_type=graph_type)

    df_relevant = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type=graph_type, prompt_type="relevant")
    df_unrelated = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type=graph_type, prompt_type="unrelated")

    if only_return_dataframes:
        return df_relevant, df_unrelated
    else:
        df = pd.concat([df_relevant, df_unrelated])
        len_df = len(df.index)
        df["new_index"] = [x for x in range(len_df)]
        df.set_index("new_index", inplace=True)
        df = df.drop("概念", axis=1)

        sns_settings()
        fig, ax = plt.subplots(1, 1, figsize=(30, 15), tight_layout=True)

        if graph_type == "suppress":
            sns.histplot(data=df, x="正解を選ぶ確率の変化率[%]", hue="凡例", bins=40, binrange=(-100, 100), ax=ax)
        else:
            sns.histplot(data=df, x="正解を選ぶ確率の変化率[%]", hue="凡例", bins=44, binrange=(-100, 1000), ax=ax)

        figure = fig.get_figure()

        save_path = os.path.join(root_save_path, f"{graph_type}_histogram.png")
        figure.savefig(save_path)
        print(f"figure is saved in {save_path}")
        plt.close()


# 品詞ごとに分割したヒストグラム
def make_df_for_pos_histogram(relevant_result_path, unrelated_result_path, *, graph_type, ground_truth_type):
    percentage_list, labels, ground_truth_list = get_datalist(relevant_result_path, unrelated_result_path, graph_type=graph_type, ground_truth_type=ground_truth_type)

    df_relevant = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type=graph_type, prompt_type="relevant")
    df_unrelated = make_dataframe_for_histogram(percentage_list, labels, ground_truth_list, graph_type=graph_type, prompt_type="unrelated")

    return df_relevant, df_unrelated


def make_histograms_comparing_pos(df_dict, root_save_path, graph_type):
    if graph_type == "suppress":
        axis_titlehead = "抑制"
        bins = 28
        binrange = (-100, 40)
    else:
        axis_titlehead = "増幅"
        bins = 44
        binrange = (-100, 1000)

    df_noun_relevant = df_dict[f"noun_{graph_type}_relevant"].replace({"凡例": {f"活性値を{axis_titlehead}し，知識ニューロンに紐づく概念のセンテンスを予測": "名詞"}})
    df_verb_adjective_adverb_relevant = df_dict[f"verb_adjective_adverb_{graph_type}_relevant"].replace({"凡例": {f"活性値を{axis_titlehead}し，知識ニューロンに紐づく概念のセンテンスを予測": "動詞・形容詞・副詞"}})

    df_relevant = pd.concat([df_noun_relevant, df_verb_adjective_adverb_relevant])
    df_relevant["new_index"] = [x for x in range(len(df_relevant.index))]
    df_relevant.set_index("new_index", inplace=True)
    df_relevant = df_relevant.drop("概念", axis=1)

    sns_settings2()
    fig, ax = plt.subplots(1, 1, figsize=(24, 10), tight_layout=True)
    sns.histplot(data=df_relevant, x="正解を選ぶ確率の変化率[%]", hue="凡例", bins=bins, binrange=binrange, ax=ax)
    figure = fig.get_figure()

    save_path = os.path.join(root_save_path, f"relevant_{graph_type}_histogram.png")
    figure.savefig(save_path)
    print(f"figure is saved in {save_path}")
    plt.close()

    # もし必要なら、ここに"unrelated"な品詞比較ヒストグラムも作る


def make_parser():
    parser = argparse.ArgumentParser(description="Make results into a graph.")

    parser.add_argument("--displayed_entity_index",
                        help="(optional) Shift the entities displayed in the graph arbitrarily.",
                        type=int, default=0
                        )
    parser.add_argument("-x", "--number_of_entities_show_in_x_axis",
                        help="Designate the number of entities displayed in the figure 3 & 4 format. Note that '16' or more may collapse the figure layout.",
                        type=int, default=15
                        )
    parser.add_argument("-rp", "--result_path", help="path for the result file (file at 'work/result/.')", required=True)
    parser.add_argument('--save_path', help="results(contain used entities) will be saved under the designed path.",
                        type=str, default='')

    return parser.parse_args()


def main():
    args = make_parser()

    # 必要であれば、以下のダウンロードスクリプトを含めて動かす
    # nltk.download('averaged_perceptron_tagger')

    # パス関連
    suppress_relevant = os.path.join(args.result_path, "suppress_activation_and_relevant_prompts.jsonl")
    suppress_unrelated = os.path.join(args.result_path, "suppress_activation_and_unrelated_prompts.jsonl")
    enhance_relevant = os.path.join(args.result_path, "enhance_activation_and_relevant_prompts.jsonl")
    enhance_unrelated = os.path.join(args.result_path, "enhance_activation_and_unrelated_prompts.jsonl")

    splited_path = args.result_path.split("/")
    dataset_type, model_name, experiment_settings = splited_path[-3], splited_path[-2], splited_path[-1]

    # グラフ保存先ディレクトリの作成
    if args.save_path:
        root_path = os.path.join(args.save_path, "figure", dataset_type, model_name, experiment_settings)
    else:
        root_path = os.path.join("../work", "figure", dataset_type, model_name, experiment_settings)
    os.makedirs(root_path, exist_ok=True)

    # figure 3, 4, 5, 6
    make_individual_concept_graph(suppress_relevant, suppress_unrelated, root_path,
                                  entity_index=args.displayed_entity_index,
                                  display_number_of_entities=args.number_of_entities_show_in_x_axis,
                                  graph_type="suppress"
                                  )
    make_individual_concept_graph(enhance_relevant, enhance_unrelated, root_path,
                                  entity_index=args.displayed_entity_index,
                                  display_number_of_entities=args.number_of_entities_show_in_x_axis,
                                  graph_type="enhance"
                                  )
    make_whole_concepts_histogram(suppress_relevant, suppress_unrelated, root_save_path=root_path, graph_type="suppress")
    make_whole_concepts_histogram(enhance_relevant, enhance_unrelated, root_save_path=root_path, graph_type="enhance")

    # preparation for figure 7, 8
    df_dict_for_pos = defaultdict()

    # 名詞の結果をdataframeに格納
    df_suppress_relevant, df_suppress_unrelated = make_df_for_pos_histogram(suppress_relevant,
                                                                            suppress_unrelated,
                                                                            graph_type="suppress",
                                                                            ground_truth_type="noun"
                                                                            )
    df_enhance_relevant, df_enhance_unrelated = make_df_for_pos_histogram(enhance_relevant,
                                                                          enhance_unrelated,
                                                                          graph_type="enhance",
                                                                          ground_truth_type="noun"
                                                                          )
    df_dict_for_pos["noun_suppress_relevant"] = df_suppress_relevant
    df_dict_for_pos["noun_suppress_unrelated"] = df_suppress_unrelated
    df_dict_for_pos["noun_enhance_relevant"] = df_enhance_relevant
    df_dict_for_pos["noun_enhance_unrelated"] = df_enhance_unrelated

    # 動詞・形容詞・形容動詞の結果をdataframeに格納
    df_suppress_relevant, df_suppress_unrelated = make_df_for_pos_histogram(suppress_relevant,
                                                                            suppress_unrelated,
                                                                            graph_type="suppress",
                                                                            ground_truth_type="verb_adjective_adverb"
                                                                            )
    df_enhance_relevant, df_enhance_unrelated = make_df_for_pos_histogram(enhance_relevant,
                                                                          enhance_unrelated,
                                                                          graph_type="enhance",
                                                                          ground_truth_type="verb_adjective_adverb"
                                                                          )
    df_dict_for_pos["verb_adjective_adverb_suppress_relevant"] = df_suppress_relevant
    df_dict_for_pos["verb_adjective_adverb_suppress_unrelated"] = df_suppress_unrelated
    df_dict_for_pos["verb_adjective_adverb_enhance_relevant"] = df_enhance_relevant
    df_dict_for_pos["verb_adjective_adverb_enhance_unrelated"] = df_enhance_unrelated

    # figure 7, 8
    make_histograms_comparing_pos(df_dict_for_pos, root_path, graph_type="suppress")
    make_histograms_comparing_pos(df_dict_for_pos, root_path, graph_type="enhance")


if __name__ == '__main__':
    main()
