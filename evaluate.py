# -*- coding: utf-8 -*-
# MIT License
# Copyright (c) 2021 Sid Black

import sys

from knowledge_neurons import KnowledgeNeurons, initialize_model_and_tokenizer, model_type
from utils.data import extract_matched_dataset_from_generics_kb
import random
import logzero
import argparse
import os
import pathlib
import numpy as np
import torch
from datasets import load_dataset
import json


def make_parser():
    parser = argparse.ArgumentParser(description='Conduct experiments')

    parser.add_argument('--seed', help='the seed of random numbers. default=42', type=int, default=42)
    parser.add_argument('-mn', '--model_name', help='the name of neural model.', type=str, default="bert-base-uncased")
    parser.add_argument('-dt', '--dataset_type', help="designate the dataset", default="generics_kb_best",
                        choices=["generics_kb_best", "generics_kb", "generics_kb_simplewiki", "generics_kb_waterloo"]
                        )
    parser.add_argument('-dp', '--dataset_path', help="the path for the GenericsKB file which manually downloaded", type=str, required=True)
    parser.add_argument('-nt', '--number_of_templates', help='the minimum number of templates which each entity have. default=4', type=int, default=4)
    parser.add_argument('--local_rank', help="local rank for multigpu processing, default=0", type=int, default=0)
    parser.add_argument('-ln', '--logfile_name', help="designate the file name of log. default='run'", type=str, default="run")
    parser.add_argument('-bs', '--batch_size', help="", type=int, default=20)
    parser.add_argument('--steps', help="number of steps in the integrated grad calculation", type=int, default=20)
    parser.add_argument('-at', '--adaptive_threshold', help="the threshold value", type=float, default=0.3)
    parser.add_argument('-sp', '--sharing_percentage', help="the threshold for the sharing percentage", type=float, default=0.5)
    parser.add_argument('-mw', '--max_words', help="the maximum number of words which each template can have", type=int, default=15)
    parser.add_argument('--save_path', help="results(contain used entities) will be saved under the designed path.",
                        type=str, default='')

    return parser.parse_args()


def make_log(args):
    log_directory = os.path.join("log", args.dataset_type)
    os.makedirs(log_directory, exist_ok=True)
    log_file_name = args.logfile_name + ".log"
    log_file_path = os.path.join(log_directory, log_file_name)
    if not os.path.isfile(log_file_path):
        log_file = pathlib.Path(log_file_path)
        log_file.touch()
    logger = logzero.setup_logger(
        logfile=log_file_path,
        disableStderrLogger=False
        )
    print('log is saved in ' + log_file_path)
    logger.info('--------start of script--------')

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logger.info('random seed is ' + str(args.seed))
    logger.info('model name is ' + args.model_name)
    logger.info('dataset type is ' + args.dataset_type)
    logger.info('the number of templates each entity at least have is ' + str(args.number_of_templates))

    return logger


def main():
    args = make_parser()

    logger = make_log(args)

    torch.cuda.set_device(args.local_rank)

    # first initialize some hyperparameters
    MODEL_NAME = args.model_name

    # these are some hyperparameters for the integrated gradients step
    BATCH_SIZE = args.batch_size
    STEPS = args.steps  # number of steps in the integrated grad calculation
    ADAPTIVE_THRESHOLD = args.adaptive_threshold  # in the paper, they find the threshold value `t` by multiplying the max attribution score by some float - this is that float.
    P = args.sharing_percentage  # the threshold for the sharing percentage

    # setup model & tokenizer
    model, tokenizer = initialize_model_and_tokenizer(MODEL_NAME)

    # initialize the knowledge neuron wrapper with your model, tokenizer and a string expressing the type of your model ('gpt2' / 'gpt_neo' / 'bert')
    kn = KnowledgeNeurons(model, tokenizer, model_type=model_type(MODEL_NAME))

    # load dataset
    dataset_kb_best = load_dataset("generics_kb", args.dataset_type, data_dir=args.dataset_path)

    # デバッグのためのコード（小さいデータで動作確認）
    #dataset_kb_best = dataset_kb_best["train"][:300]
    #dataset_kb_best = dataset_kb_best["train"][:608347]

    dataset_kb_best = dataset_kb_best["train"]

    # make dataset by the conditions
    matched_dataset = extract_matched_dataset_from_generics_kb(
        dataset_kb_best,
        tokenizer,
        num_of_templates=args.number_of_templates,
        max_words=args.max_words,
    )

    logger.info('Number of entities covered this time: ' + str(len(matched_dataset.keys())))
    logger.info('')


    # 結果の保存先ディレクトリやファイル名を指定するための変数
    # 使用したデータセットとモデルの名前を連結した文字列
    if "/" in args.model_name:
        model_name = args.model_name.split("/")[1]
    else:
        model_name = args.model_name
    dataset_and_model = os.path.join(args.dataset_type, model_name)
    # 実験のハイパラを連結した文字列
    experiment_settings = f"nt_{args.number_of_templates}_at_{args.adaptive_threshold}_mw_{args.max_words}"

    # Write out the entities used as the dataset in this condition to a file.
    if matched_dataset:
        ground_truths = []
        for entity in matched_dataset.keys():
            ground_truths.append(entity)
        dic = {"concept": ground_truths}

        if args.save_path:
            entities_path = os.path.join(args.save_path.rstrip("/"), "entities", dataset_and_model)
        else:
            entities_path = os.path.join("work", "entities", dataset_and_model)
        os.makedirs(entities_path, exist_ok=True)

        save_entities_path = os.path.join(entities_path, f"{experiment_settings}.txt")
        with open(save_entities_path, mode="w") as fi:
            json.dump(dic, fi, indent=4)


    # experiment
    total_templates = 0
    total_refined_neurons = 0
    suppress_relevant_results = []
    suppress_unrelated_results = []
    enhance_relevant_results = []
    enhance_unrelated_results = []

    unrelated_prompt = "[MASK] is the official language of the solomon islands"
    unrelated_ground_truth = "english"

    for entity, templates in matched_dataset.items():
        TEXTS = list(templates)
        TEXT = TEXTS[0]
        GROUND_TRUTH = entity

        logger.info("Ground Truth: " + GROUND_TRUTH)
        logger.info('The number of related templates: ' + str(len(TEXTS)))
        logger.info(f'Templates: {TEXTS}')
        logger.info("")

        total_templates += len(TEXTS)

        # use the integrated gradients technique to find some refined neurons for your set of prompts
        refined_neurons = kn.get_refined_neurons(
            TEXTS,
            GROUND_TRUTH,
            p=P,
            batch_size=BATCH_SIZE,
            steps=STEPS,
            coarse_adaptive_threshold=ADAPTIVE_THRESHOLD,
        )

        logger.info('refining done')
        number_of_refined_neurons = len(refined_neurons)
        total_refined_neurons += number_of_refined_neurons

        # suppress the activations at the refined neurons + test the effect on a relevant prompt
        # 'results_dict' is a dictionary containing the probability of the ground truth being generated before + after modification, as well as other info
        # 'unpatch_fn' is a function you can use to undo the activation suppression in the model.
        # By default, the suppression is removed at the end of any function that applies a patch, but you can set 'undo_modification=False',
        # run your own experiments with the activations / weights still modified, then run 'unpatch_fn' to undo the modifications
        logger.info('suppress the activations at the refined neurons + test the effect on a relevant prompt')
        results_dict, unpatch_fn = kn.suppress_knowledge(
            TEXT, GROUND_TRUTH, refined_neurons
        )
        suppress_relevant_results.append(
            {
                "ground_truth": GROUND_TRUTH,
                "number_of_refined_neurons": number_of_refined_neurons,
                "ground_truth_prob_before_modifying": results_dict["before"]["gt_prob"],
                "argmax_entity_before_modifying": results_dict["before"]["argmax_completion"],
                "argmax_entity_prob_before_modifying": results_dict["before"]["argmax_prob"],
                "ground_truth_prob_after_modifying": results_dict["after"]["gt_prob"],
                "argmax_entity_after_modifying": results_dict["after"]["argmax_completion"],
                "argmax_entity_prob_after_modifying": results_dict["after"]["argmax_prob"],
                "sentence": TEXT
            }
        )

        # suppress the activations at the refined neurons + test the effect on an unrelated prompt
        logger.info('suppress the activations at the refined neurons + test the effect on an unrelated prompt')
        results_dict, unpatch_fn = kn.suppress_knowledge(
            unrelated_prompt,
            unrelated_ground_truth,
            refined_neurons,
        )
        suppress_unrelated_results.append(
            {
                "ground_truth": GROUND_TRUTH,
                "unrelated_ground_truth": unrelated_ground_truth,
                "number_of_refined_neurons": number_of_refined_neurons,
                "ground_truth_prob_before_modifying": results_dict["before"]["gt_prob"],
                "argmax_entity_before_modifying": results_dict["before"]["argmax_completion"],
                "argmax_entity_prob_before_modifying": results_dict["before"]["argmax_prob"],
                "ground_truth_prob_after_modifying": results_dict["after"]["gt_prob"],
                "argmax_entity_after_modifying": results_dict["after"]["argmax_completion"],
                "argmax_entity_prob_after_modifying": results_dict["after"]["argmax_prob"],
                "sentence": TEXT,
                "unrelated_prompt": unrelated_prompt
            }
        )

        # enhance the activations at the refined neurons + test the effect on a relevant prompt
        logger.info('enhance the activations at the refined neurons + test the effect on a relevant prompt')
        results_dict, unpatch_fn = kn.enhance_knowledge(TEXT, GROUND_TRUTH, refined_neurons)
        enhance_relevant_results.append(
            {
                "ground_truth": GROUND_TRUTH,
                "number_of_refined_neurons": number_of_refined_neurons,
                "ground_truth_prob_before_modifying": results_dict["before"]["gt_prob"],
                "argmax_entity_before_modifying": results_dict["before"]["argmax_completion"],
                "argmax_entity_prob_before_modifying": results_dict["before"]["argmax_prob"],
                "ground_truth_prob_after_modifying": results_dict["after"]["gt_prob"],
                "argmax_entity_after_modifying": results_dict["after"]["argmax_completion"],
                "argmax_entity_prob_after_modifying": results_dict["after"]["argmax_prob"],
                "sentence": TEXT
            }
        )

        # enhance the activations at the refined neurons + test the effect on an unrelated prompt
        logger.info('enhance the activations at the refined neurons + test the effect on an unrelated prompt')
        results_dict, unpatch_fn = kn.enhance_knowledge(
            unrelated_prompt,
            unrelated_ground_truth,
            refined_neurons,
        )
        enhance_unrelated_results.append(
            {
                "ground_truth": GROUND_TRUTH,
                "unrelated_ground_truth": unrelated_ground_truth,
                "number_of_refined_neurons": number_of_refined_neurons,
                "ground_truth_prob_before_modifying": results_dict["before"]["gt_prob"],
                "argmax_entity_before_modifying": results_dict["before"]["argmax_completion"],
                "argmax_entity_prob_before_modifying": results_dict["before"]["argmax_prob"],
                "ground_truth_prob_after_modifying": results_dict["after"]["gt_prob"],
                "argmax_entity_after_modifying": results_dict["after"]["argmax_completion"],
                "argmax_entity_prob_after_modifying": results_dict["after"]["argmax_prob"],
                "sentence": TEXT,
                "unrelated_prompt": unrelated_prompt
            }
        )

        logger.debug('')


    # path for save the results
    if args.save_path:
        result_path = os.path.join(args.save_path.rstrip("/"), "result", dataset_and_model, experiment_settings)
    else:
        result_path = os.path.join("work", "result", dataset_and_model, experiment_settings)
    os.makedirs(result_path, exist_ok=True)
    suppress_relevant = os.path.join(result_path, "suppress_activation_and_relevant_prompts.jsonl")
    suppress_unrelated = os.path.join(result_path, "suppress_activation_and_unrelated_prompts.jsonl")
    enhance_relevant = os.path.join(result_path, "enhance_activation_and_relevant_prompts.jsonl")
    enhance_unrelated = os.path.join(result_path, "enhance_activation_and_unrelated_prompts.jsonl")
    other_information = os.path.join(result_path, "other_information.txt")

    with open(suppress_relevant, mode="w", encoding='utf-8') as sr_fi, \
         open(suppress_unrelated, mode="w", encoding='utf-8') as su_fi, \
         open(enhance_relevant, mode="w", encoding='utf-8') as er_fi,\
         open(enhance_unrelated, mode="w", encoding='utf-8') as eu_fi, \
         open(other_information, mode="w") as oi_fi:
        result_file_pointers = [sr_fi, su_fi, er_fi, eu_fi]
        results = [suppress_relevant_results, suppress_unrelated_results, enhance_relevant_results, enhance_unrelated_results]
        for fp, result in zip(result_file_pointers, results):
            for record in result:
                json.dump(record, fp, ensure_ascii=False)
                fp.write('\n')

        try:
            logger.info(f"average templates per GROUND_TRUTH = {total_templates / len(matched_dataset)}")
            oi_fi.write(f"average templates per GROUND_TRUTH = {total_templates / len(matched_dataset)}\n")
            logger.info(f"average refined neurons per GROUND_TRUTH = {total_refined_neurons / len(matched_dataset)}")
            oi_fi.write(f"average refined neurons per GROUND_TRUTH = {total_refined_neurons / len(matched_dataset)}")
        except ZeroDivisionError as e:
            print(f"Error: {e}")
            sys.exit(1)

    logger.debug('script done!')
    logger.debug('')


if __name__ == '__main__':
    main()
