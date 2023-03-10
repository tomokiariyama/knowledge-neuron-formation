from tqdm import tqdm
from collections import defaultdict
import json
from logzero import logger
import sys
import re
import unicodedata
from transformers import BertTokenizer


def extract_raw_dataset_from_jsonlines(file_path):
    """
    The function which returns the dictionary made from {ConceptNet, TREx, Google_RE or Squad} dataset and whose keys are 'subject, object, template'.
    """

    with open(file_path) as fi:
        dataset_list = []
        for line in tqdm(fi):
            d = defaultdict(str)

            case = json.loads(line)
            try:
                d["sub"] = case["sub_surface"]
            except KeyError:
                try:
                    d["sub"] = case["sub_label"]
                except KeyError:
                    try:
                        d["sub"] = case["sub"]
                    except KeyError:
                        logger.info(f"filepath: {file_path}")
                        logger.info(f"case: {case}")
                        logger.error("There is no key corresponding to 'subject' in this dataset.")
                        sys.exit(1)

            try:
                d["obj"] = case["obj_surface"]
            except KeyError:
                try:
                    d["obj"] = case["obj_label"]
                except KeyError:
                    try:
                        d["obj"] = case["obj"]
                    except KeyError:
                        logger.info(f"filepath: {file_path}")
                        logger.info(f"case: {case}")
                        logger.error("There is no key corresponding to 'object' in this dataset.")
                        sys.exit(1)

            try:
                for masked_sentence in case["masked_sentences"]:
                    if masked_sentence.count("[MASK]") == 1:
                        d["masked_sentence"] = masked_sentence
                # If we couldn't find the masked_sentence that has only one [MASK] token, skip that case.
                if not d["masked_sentence"]:
                    continue
            except KeyError:
                try:
                    d["masked_sentence"] = case["evidences"][0]["masked_sentence"]
                except KeyError:
                    logger.info(f"filepath: {file_path}")
                    logger.info(f"case: {case}")
                    logger.error("There is no 'masked_sentence' key in this dataset.")
                    sys.exit(1)

            dataset_list.append(d)

    return dataset_list


def extract_matched_dataset(dataset_list, entity_type, num_of_templates, max_words, is_remove_unk_concept):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")

    if entity_type == "subject":
        d = defaultdict(set)  # key: concept(str), value: set consists of concept's templates

        # At first, replace "[MASK]" to obj, sub to "[MASK]" in the template.
        # Then, append templates as set type to the dictionary whose key is sub.
        # The reason for using the set type is that some templates are the same but have different obj, but if the template is the same, it is counted as one.
        for case in tqdm(dataset_list):
            try:
                r = re.compile(f'{case["sub"]}',
                               re.IGNORECASE)  # Replace sub to "[MASK]" without considering the case of the first letter of sub.
            except re.error:
                logger.warning(f'skipped a case which has sub_label: {case["sub"]}')
                continue

            # Since the original masked_sentence may not contain an exact match between the [MASK] token and the sub to be converted, this case is excluded.
            if not re.search(r, case["masked_sentence"]):
                logger.warning(
                    f'skipped a case which has masked_sentence with no sub_label, sub_label: {case["sub"]}, masked_sentence: {case["masked_sentence"]}')
                continue

            no_mask_sentence = case["masked_sentence"].replace('[MASK]', case["obj"])
            new_masked_sentence = re.sub(r, '[MASK]', no_mask_sentence,
                                         1)  # Make sure that only one mask token appears in a new_masked_sentence.
            new_masked_sentence = unicodedata.normalize("NFKD",
                                                        new_masked_sentence)  # Replace the Unicode's no-break-space and so on.

            # Restrict the number of maximum words in a template.
            if len(new_masked_sentence.split(" ")) <= max_words:
                d[case["sub"]].add(new_masked_sentence)
            else:
                continue

        # Exclude subject entities that do not meet the default number of templates.
        delete_entities = []
        if is_remove_unk_concept:
            for sub in d.keys():
                if len(d[sub]) < num_of_templates or tokenizer.convert_tokens_to_ids(sub) == unk_id:
                    delete_entities.append(sub)
        else:
            for sub in d.keys():
                if len(d[sub]) < num_of_templates:
                    delete_entities.append(sub)
        for delete_key in delete_entities:
            del d[delete_key]

        return d

    elif entity_type == "object":
        d = defaultdict(set)

        # Register a template for each object entity in the dictionary as a set type
        for case in tqdm(dataset_list):
            # Restrict the number of maximum words in a template.
            if len(case["masked_sentence"].split(" ")) <= max_words:
                d[case["obj"]].add(case["masked_sentence"])
            else:
                continue

        # Exclude object entities that do not meet the default number of templates.
        delete_entities = []
        if is_remove_unk_concept:
            for obj in d.keys():
                if len(d[obj]) < num_of_templates or tokenizer.convert_tokens_to_ids(obj) == unk_id:
                    delete_entities.append(obj)
        else:
            for obj in d.keys():
                if len(d[obj]) < num_of_templates:
                    delete_entities.append(obj)
        for delete_key in delete_entities:
            del d[delete_key]

        return d

    else:
        try:
            raise ValueError("entity type is somewhat wrong")
        except ValueError as e:
            print(e)
        sys.exit(1)


def extract_matched_dataset_from_generics_kb(dataset, tokenizer, *, num_of_templates, max_words):
    """
    dataset: {'source': Value(dtype='string', id=None),
              'term': Value(dtype='string', id=None),
              'quantifier_frequency': Value(dtype='string', id=None),
              'quantifier_number': Value(dtype='string', id=None),
              'generic_sentence': Value(dtype='string', id=None),
              'score': Value(dtype='float64', id=None)
              }
        -> dict{'term': [generic_sentences]}
        (= dict{'concept': [templates]})
    """

    ids = range(tokenizer.vocab_size)
    lm_vocab = set(tokenizer.convert_ids_to_tokens(ids))
    # lm_vocab = set(tokenizer.vocab.keys())

    d = defaultdict(set)

    # 言語モデルの語彙辞書に載っている"term"（＝概念）の"generic_sentence"（単語数はmax_words以下）を取ってくる
    for term, generic_sentence in zip(tqdm(dataset["term"]), dataset["generic_sentence"]):
        if term in lm_vocab and len(generic_sentence.split(" ")) <= max_words:
            d[term].add(generic_sentence.lower())

    # 各テンプレートをマスク
    for concept, templates in tqdm(d.items()):
        masked_templates = set()
        for template in templates:
            tokenized_template = tokenizer.tokenize(template)
            masked_idx_list = [i for i, x in enumerate(tokenized_template) if x == concept]
            # テンプレートにconceptが一つも入っていない(ex) concept="carry", テンプレート="Sound carries well over water.")、
            # または二つ以上入ってしまっている場合は取り除く
            # TODO: できれば、活用形も判定できるようにしたい
            if len(masked_idx_list) != 1:
                print(f"skipped: concept='{concept}', template='{template}'")
                continue
            tokenized_template[masked_idx_list[0]] = "[MASK]"
            masked_templates.add(" ".join(tokenized_template))
        d[concept] = masked_templates

    # 一概念あたりのテンプレート数が基準に満たない概念を削除
    delete_entities = []
    for concept in d.keys():
        if len(d[concept]) < num_of_templates:
            delete_entities.append(concept)
    for delete_key in delete_entities:
        del d[delete_key]

    return d
