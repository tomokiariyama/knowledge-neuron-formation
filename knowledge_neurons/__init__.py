# MIT License
# Copyright (c) 2021 Sid Black

from transformers import (
    BertTokenizer,
    BertLMHeadModel,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)
from .knowledge_neurons import KnowledgeNeurons
from .data import pararel, pararel_expanded, PARAREL_RELATION_NAMES

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-uncased"]
GPT2_MODELS = ["gpt2"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
MBERT_MODELS = [
    "google/multiberts-seed_0-step_0k",
    "google/multiberts-seed_0-step_20k",
    "google/multiberts-seed_0-step_40k",
    "google/multiberts-seed_0-step_60k",
    "google/multiberts-seed_0-step_80k",
    "google/multiberts-seed_0-step_100k",
    "google/multiberts-seed_0-step_120k",
    "google/multiberts-seed_0-step_140k",
    "google/multiberts-seed_0-step_160k",
    "google/multiberts-seed_0-step_180k",
    "google/multiberts-seed_0-step_200k",
    "google/multiberts-seed_0-step_300k",
    "google/multiberts-seed_0-step_400k",
    "google/multiberts-seed_0-step_500k",
    "google/multiberts-seed_0-step_600k",
    "google/multiberts-seed_0-step_700k",
    "google/multiberts-seed_0-step_800k",
    "google/multiberts-seed_0-step_900k",
    "google/multiberts-seed_0-step_1000k",
    "google/multiberts-seed_0-step_1100k",
    "google/multiberts-seed_0-step_1200k",
    "google/multiberts-seed_0-step_1300k",
    "google/multiberts-seed_0-step_1400k",
    "google/multiberts-seed_0-step_1500k",
    "google/multiberts-seed_0-step_1600k",
    "google/multiberts-seed_0-step_1700k",
    "google/multiberts-seed_0-step_1800k",
    "google/multiberts-seed_0-step_1900k",
    "google/multiberts-seed_0-step_2000k"
]
ALL_MODELS = BERT_MODELS + MBERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS


def initialize_model_and_tokenizer(model_name: str):
    if model_name in (BERT_MODELS + MBERT_MODELS):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in GPT_NEO_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")

    model.eval()

    return model, tokenizer


def model_type(model_name: str):
    if model_name in (BERT_MODELS + MBERT_MODELS):
        return "bert"
    elif model_name in GPT2_MODELS:
        return "gpt2"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    else:
        raise ValueError(f"Model {model_name} not supported")
